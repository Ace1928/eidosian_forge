import flask
from flask import make_response
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import utils
from keystone.conf import CONF
from keystone import exception
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
Get an OAuth2.0 certificate-bound Access Token.