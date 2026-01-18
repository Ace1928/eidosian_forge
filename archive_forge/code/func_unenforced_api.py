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
def unenforced_api(f):
    """Decorate a resource method to mark is as an unenforced API.

    Explicitly exempts an API from receiving the enforced API check,
    specifically for cases such as user self-service password changes (or
    other APIs that must work without already having a token).

    This decorator may also be used if the API has extended enforcement
    logic/varying enforcement logic (such as some of the AUTH paths) where
    the full enforcement will be implemented directly within the methods.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        set_unenforced_ok()
        return f(*args, **kwargs)
    return wrapper