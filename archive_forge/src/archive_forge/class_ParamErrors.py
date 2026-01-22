from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
class ParamErrors(Test):

    @cherrypy.expose
    def one_positional(self, param1):
        return 'data'

    @cherrypy.expose
    def one_positional_args(self, param1, *args):
        return 'data'

    @cherrypy.expose
    def one_positional_args_kwargs(self, param1, *args, **kwargs):
        return 'data'

    @cherrypy.expose
    def one_positional_kwargs(self, param1, **kwargs):
        return 'data'

    @cherrypy.expose
    def no_positional(self):
        return 'data'

    @cherrypy.expose
    def no_positional_args(self, *args):
        return 'data'

    @cherrypy.expose
    def no_positional_args_kwargs(self, *args, **kwargs):
        return 'data'

    @cherrypy.expose
    def no_positional_kwargs(self, **kwargs):
        return 'data'
    callable_object = ParamErrorsCallable()

    @cherrypy.expose
    def raise_type_error(self, **kwargs):
        raise TypeError('Client Error')

    @cherrypy.expose
    def raise_type_error_with_default_param(self, x, y=None):
        return '%d' % 'a'

    @cherrypy.expose
    @handler_dec
    def raise_type_error_decorated(self, *args, **kwargs):
        raise TypeError('Client Error')