import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
class APIBase(object, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def _name(self):
        """Override with an attr consisting of the API Name, e.g 'users'."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _import_name(self):
        """Override with an attr consisting of the value of `__name__`."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def resource_mapping(self):
        """An attr containing of an iterable of :class:`ResourceMap`.

        Each :class:`ResourceMap` is a NamedTuple with the following elements:

            * resource: a :class:`flask_restful.Resource` class or subclass

            * url: a url route to match for the resource, standard flask
                   routing rules apply. Any url variables will be passed
                   to the resource method as args. (str)

            * alternate_urls: an iterable of url routes to match for the
                              resource, standard flask routing rules apply.
                              These rules are in addition (for API compat) to
                              the primary url. Any url variables will be
                              passed to the resource method as args. (iterable)

            * json_home_data: :class:`JsonHomeData` populated with relevant
                              info for populated JSON Home Documents or None.

            * kwargs: a dict of optional value(s) that can further modify the
                      handling of the routing.

                      * endpoint: endpoint name (defaults to
                                  :meth:`Resource.__name__.lower`
                                  Can be used to reference this route in
                                  :class:`fields.Url` fields (str)

                      * resource_class_args: args to be forwarded to the
                                             constructor of the resource.
                                             (tuple)

                      * resource_class_kwargs: kwargs to be forwarded to the
                                               constructor of the resource.
                                               (dict)

                      Additional keyword arguments not specified above will be
                      passed as-is to :meth:`flask.Flask.add_url_rule`.
        """
        raise NotImplementedError()

    @property
    def resources(self):
        return []

    @staticmethod
    def _build_bp_url_prefix(prefix):
        parts = ['/v3']
        if prefix:
            parts.append(prefix.lstrip('/'))
        return '/'.join(parts).rstrip('/')

    @property
    def api(self):
        return self.__api

    @property
    def blueprint(self):
        return self.__blueprint

    def __init__(self, blueprint_url_prefix='', api_url_prefix='', default_mediatype='application/json', decorators=None, errors=None):
        self.__before_request_functions_added = False
        self.__after_request_functions_added = False
        self._default_mediatype = default_mediatype
        blueprint_url_prefix = blueprint_url_prefix.rstrip('/')
        api_url_prefix = api_url_prefix.rstrip('/')
        if api_url_prefix and (not api_url_prefix.startswith('/')):
            self._api_url_prefix = '/%s' % api_url_prefix
        else:
            self._api_url_prefix = api_url_prefix or getattr(self, '_api_url_prefix', '')
        if blueprint_url_prefix and (not blueprint_url_prefix.startswith('/')):
            self._blueprint_url_prefix = self._build_bp_url_prefix('/%s' % blueprint_url_prefix)
        else:
            self._blueprint_url_prefix = self._build_bp_url_prefix(blueprint_url_prefix)
        self.__blueprint = blueprints.Blueprint(name=self._name, import_name=self._import_name, url_prefix=self._blueprint_url_prefix)
        self.__api = flask_restful.Api(app=self.__blueprint, prefix=self._api_url_prefix, default_mediatype=self._default_mediatype, decorators=decorators, errors=errors)
        self.__api.representation('application/json')(self._output_json)
        self._add_resources()
        self._add_mapped_resources()
        self._register_before_request_functions()
        self._register_after_request_functions()
        msg = '%s_request functions not registered'
        assert self.__before_request_functions_added, msg % 'before'
        assert self.__after_request_functions_added, msg % 'after'

    def _add_resources(self):
        for r in self.resources:
            c_key = getattr(r, 'collection_key', None)
            m_key = getattr(r, 'member_key', None)
            r_pfx = getattr(r, 'api_prefix', None)
            if not c_key or not m_key:
                LOG.debug('Unable to add resource %(resource)s to API %(name)s, both `member_key` and `collection_key` must be implemented. [collection_key(%(col_key)s) member_key(%(m_key)s)]', {'resource': r.__name__, 'name': self._name, 'col_key': c_key, 'm_key': m_key})
                continue
            if r_pfx != self._api_url_prefix:
                LOG.debug('Unable to add resource %(resource)s to API as the API Prefixes do not match: %(apfx)r != %(rpfx)r', {'resource': r.__name__, 'rpfx': r_pfx, 'apfx': self._api_url_prefix})
                continue
            collection_path = '/%s' % c_key
            if getattr(r, '_id_path_param_name_override', None):
                member_id_key = getattr(r, '_id_path_param_name_override')
            else:
                member_id_key = '%(member_key)s_id' % {'member_key': m_key}
            entity_path = '/%(collection)s/<string:%(member)s>' % {'collection': c_key, 'member': member_id_key}
            jh_e_path = _URL_SUBST.sub('{\\1}', '%(pfx)s/%(e_path)s' % {'pfx': self._api_url_prefix, 'e_path': entity_path.lstrip('/')})
            LOG.debug('Adding standard routes to API %(name)s for `%(resource)s` (API Prefix: %(prefix)s) [%(collection_path)s, %(entity_path)s]', {'name': self._name, 'resource': r.__class__.__name__, 'collection_path': collection_path, 'entity_path': entity_path, 'prefix': self._api_url_prefix})
            self.api.add_resource(r, collection_path, entity_path)
            resource_rel_func = getattr(r, 'json_home_resource_rel_func', json_home.build_v3_resource_relation)
            resource_rel_status = getattr(r, 'json_home_resource_status', None)
            collection_rel_resource_name = getattr(r, 'json_home_collection_resource_name_override', c_key)
            collection_rel = resource_rel_func(resource_name=collection_rel_resource_name)
            href_val = '%(pfx)s%(collection_path)s' % {'pfx': self._api_url_prefix, 'collection_path': collection_path}
            additional_params = getattr(r, 'json_home_additional_parameters', {})
            if additional_params:
                rel_data = dict()
                rel_data['href-template'] = _URL_SUBST.sub('{\\1}', href_val)
                rel_data['href-vars'] = additional_params
            else:
                rel_data = {'href': href_val}
            member_rel_resource_name = getattr(r, 'json_home_member_resource_name_override', m_key)
            entity_rel = resource_rel_func(resource_name=member_rel_resource_name)
            id_str = member_id_key
            parameter_rel_func = getattr(r, 'json_home_parameter_rel_func', json_home.build_v3_parameter_relation)
            id_param_rel = parameter_rel_func(parameter_name=id_str)
            entity_rel_data = {'href-template': jh_e_path, 'href-vars': {id_str: id_param_rel}}
            if additional_params:
                entity_rel_data.setdefault('href-vars', {}).update(additional_params)
            if resource_rel_status is not None:
                json_home.Status.update_resource_data(rel_data, resource_rel_status)
                json_home.Status.update_resource_data(entity_rel_data, resource_rel_status)
            json_home.JsonHomeResources.append_resource(collection_rel, rel_data)
            json_home.JsonHomeResources.append_resource(entity_rel, entity_rel_data)

    def _add_mapped_resources(self):
        for r in self.resource_mapping:
            alt_url_json_home_data = []
            LOG.debug('Adding resource routes to API %(name)s: [%(url)r %(kwargs)r]', {'name': self._name, 'url': r.url, 'kwargs': r.kwargs})
            urls = [r.url]
            if r.alternate_urls is not None:
                for element in r.alternate_urls:
                    if self._api_url_prefix:
                        LOG.debug('Unable to add additional resource route `%(route)s` to API %(name)s because API has a URL prefix. Only APIs without explicit prefixes can have alternate URL routes added.', {'route': element['url'], 'name': self._name})
                        continue
                    LOG.debug('Adding additional resource route (alternate) to API %(name)s: [%(url)r %(kwargs)r]', {'name': self._name, 'url': element['url'], 'kwargs': r.kwargs})
                    urls.append(element['url'])
                    if element.get('json_home'):
                        alt_url_json_home_data.append(element['json_home'])
            self.api.add_resource(r.resource, *urls, **r.kwargs)
            if r.json_home_data:
                resource_data = {}
                conv_url = '%(pfx)s/%(url)s' % {'url': _URL_SUBST.sub('{\\1}', r.url).lstrip('/'), 'pfx': self._api_url_prefix}
                if r.json_home_data.path_vars:
                    resource_data['href-template'] = conv_url
                    resource_data['href-vars'] = r.json_home_data.path_vars
                else:
                    resource_data['href'] = conv_url
                json_home.Status.update_resource_data(resource_data, r.json_home_data.status)
                json_home.JsonHomeResources.append_resource(r.json_home_data.rel, resource_data)
                for element in alt_url_json_home_data:
                    json_home.JsonHomeResources.append_resource(element.rel, resource_data)

    def _register_before_request_functions(self, functions=None):
        """Register functions to be executed in the `before request` phase.

        Override this method and pass in via "super" any additional functions
        that should be registered. It is assumed that any override will also
        accept a "functions" list and append the passed in values to it's
        list prior to calling super.

        Each function will be called with no arguments and expects a NoneType
        return. If the function returns a value, that value will be returned
        as the response to the entire request, no further processing will
        happen.

        :param functions: list of functions that will be run in the
                          `before_request` phase.
        :type functions: list
        """
        functions = functions or []
        msg = 'before_request functions already registered'
        assert not self.__before_request_functions_added, msg
        self.__blueprint.before_request(_initialize_rbac_enforcement_check)
        for f in functions:
            self.__blueprint.before_request(f)
        self.__before_request_functions_added = True

    def _register_after_request_functions(self, functions=None):
        """Register functions to be executed in the `after request` phase.

        Override this method and pass in via "super" any additional functions
        that should be registered. It is assumed that any override will also
        accept a "functions" list and append the passed in values to it's
        list prior to calling super.

        Each function will be called with a single argument of the Response
        class type. The function must return either the passed in Response or
        a new Response. NOTE: As of flask 0.7, these functions may not be
        executed in the case of an unhandled exception.

        :param functions: list of functions that will be run in the
                          `after_request` phase.
        :type functions: list
        """
        functions = functions or []
        msg = 'after_request functions already registered'
        assert not self.__after_request_functions_added, msg
        self.__blueprint.after_request(_assert_rbac_enforcement_called)
        self.__blueprint.after_request(_remove_content_type_on_204)
        for f in functions:
            self.__blueprint.after_request(f)
        self.__after_request_functions_added = True

    @staticmethod
    def _output_json(data, code, headers=None):
        """Make a Flask response with a JSON encoded body.

        This is a replacement of the default that is shipped with flask-RESTful
        as we need oslo_serialization for the wider datatypes in our objects
        that are serialized to json.
        """
        settings = flask.current_app.config.get('RESTFUL_JSON', {})
        if flask.current_app.debug:
            settings.setdefault('indent', 4)
            settings.setdefault('sort_keys', not flask_restful.utils.PY3)
        dumped = jsonutils.dumps(data, **settings) + '\n'
        resp = flask.make_response(dumped, code)
        resp.headers.extend(headers or {})
        return resp

    @classmethod
    def instantiate_and_register_to_app(cls, flask_app):
        """Build the API object and register to the passed in flask_app.

        This is a simplistic loader that makes assumptions about how the
        blueprint is loaded. Anything beyond defaults should be done
        explicitly via normal instantiation where more values may be passed
        via :meth:`__init__`.

        :returns: :class:`keystone.server.flask.common.APIBase`
        """
        inst = cls()
        flask_app.register_blueprint(inst.blueprint)
        return inst