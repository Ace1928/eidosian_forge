import functools
import sys
import flask
import oslo_i18n
from oslo_log import log
from oslo_middleware import healthcheck
import keystone.api
from keystone import exception
from keystone.oauth2 import handlers as oauth2_handlers
from keystone.receipt import handlers as receipt_handlers
from keystone.server.flask import common as ks_flask
from keystone.server.flask.request_processing import json_body
from keystone.server.flask.request_processing import req_logging
def _best_match_language():
    """Determine the best available locale.

    This returns best available locale based on the Accept-Language HTTP
    header passed in the request.
    """
    if not flask.request.accept_languages:
        return None
    return flask.request.accept_languages.best_match(oslo_i18n.get_available_languages('keystone'))