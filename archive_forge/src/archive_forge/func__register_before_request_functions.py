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