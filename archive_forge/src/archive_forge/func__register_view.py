from __future__ import absolute_import
from functools import wraps, partial
from flask import request, url_for, current_app
from flask import abort as original_flask_abort
from flask import make_response as original_flask_make_response
from flask.views import MethodView
from flask.signals import got_request_exception
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException, MethodNotAllowed, NotFound, NotAcceptable, InternalServerError
from werkzeug.wrappers import Response as ResponseBase
from flask_restful.utils import http_status_message, unpack, OrderedDict
from flask_restful.representations.json import output_json
import sys
from types import MethodType
import operator
def _register_view(self, app, resource, *urls, **kwargs):
    endpoint = kwargs.pop('endpoint', None) or resource.__name__.lower()
    self.endpoints.add(endpoint)
    resource_class_args = kwargs.pop('resource_class_args', ())
    resource_class_kwargs = kwargs.pop('resource_class_kwargs', {})
    if endpoint in getattr(app, 'view_functions', {}):
        previous_view_class = app.view_functions[endpoint].__dict__['view_class']
        if previous_view_class != resource:
            raise ValueError('This endpoint (%s) is already set to the class %s.' % (endpoint, previous_view_class.__name__))
    resource.mediatypes = self.mediatypes_method()
    resource.endpoint = endpoint
    resource_func = self.output(resource.as_view(endpoint, *resource_class_args, **resource_class_kwargs))
    for decorator in self.decorators:
        resource_func = decorator(resource_func)
    for url in urls:
        if self.blueprint:
            if self.blueprint_setup:
                self.blueprint_setup.add_url_rule(url, view_func=resource_func, **kwargs)
                continue
            else:
                rule = partial(self._complete_url, url)
        else:
            rule = self._complete_url(url, '')
        app.add_url_rule(rule, view_func=resource_func, **kwargs)