import abc
import collections
import inspect
import itertools
import operator
import typing as ty
import urllib.parse
import warnings
import jsonpatch
from keystoneauth1 import adapter
from keystoneauth1 import discover
from requests import structures
from openstack import _log
from openstack import exceptions
from openstack import format
from openstack import utils
from openstack import warnings as os_warnings
class QueryParameters:

    def __init__(self, *names, include_pagination_defaults=True, **mappings):
        """Create a dict of accepted query parameters

        :param names: List of strings containing client-side query parameter
            names. Each name in the list maps directly to the name
            expected by the server.
        :param mappings: Key-value pairs where the key is the client-side
            name we'll accept here and the value is the name
            the server expects, e.g, ``changes_since=changes-since``.
            Additionally, a value can be a dict with optional keys:

            - ``name`` - server-side name,
            - ``type`` - callable to convert from client to server
              representation
        :param include_pagination_defaults: If true, include default pagination
            parameters, ``limit`` and ``marker``. These are the most common
            query parameters used for listing resources in OpenStack APIs.
        """
        self._mapping: ty.Dict[str, ty.Union[str, ty.Dict]] = {}
        if include_pagination_defaults:
            self._mapping.update({'limit': 'limit', 'marker': 'marker'})
        self._mapping.update({name: name for name in names})
        self._mapping.update(mappings)

    def _validate(self, query, base_path=None, allow_unknown_params=False):
        """Check that supplied query keys match known query mappings

        :param dict query: Collection of key-value pairs where each key is the
            client-side parameter name or server side name.
        :param base_path: Formatted python string of the base url path for
            the resource.
        :param allow_unknown_params: Exclude query params not known by the
            resource.

        :returns: Filtered collection of the supported QueryParameters
        """
        expected_params = list(self._mapping)
        expected_params.extend((value.get('name', key) if isinstance(value, dict) else value for key, value in self._mapping.items()))
        if base_path:
            expected_params += utils.get_string_format_keys(base_path)
        invalid_keys = set(query) - set(expected_params)
        if not invalid_keys:
            return query
        elif not allow_unknown_params:
            raise exceptions.InvalidResourceQuery(message='Invalid query params: %s' % ','.join(invalid_keys), extra_data=invalid_keys)
        else:
            known_keys = set(query).intersection(set(expected_params))
            return {k: query[k] for k in known_keys}

    def _transpose(self, query, resource_type):
        """Transpose the keys in query based on the mapping

        If a query is supplied with its server side name, we will still use
        it, but take preference to the client-side name when both are supplied.

        :param dict query: Collection of key-value pairs where each key is the
                           client-side parameter name to be transposed to its
                           server side name.
        :param resource_type: Class of a resource.
        """
        result = {}
        for client_side, server_side in self._mapping.items():
            if isinstance(server_side, dict):
                name = server_side.get('name', client_side)
                type_ = server_side.get('type')
            else:
                name = server_side
                type_ = None
            try:
                provide_resource_type = len(inspect.getfullargspec(type_).args) > 1
            except TypeError:
                provide_resource_type = False
            if client_side in query:
                value = query[client_side]
            elif name in query:
                value = query[name]
            else:
                continue
            if type_ is not None:
                if provide_resource_type:
                    result[name] = type_(value, resource_type)
                else:
                    result[name] = type_(value)
            else:
                result[name] = value
        return result