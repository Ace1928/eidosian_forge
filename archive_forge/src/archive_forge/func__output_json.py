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