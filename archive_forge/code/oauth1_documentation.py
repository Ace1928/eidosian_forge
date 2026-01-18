import flask
from oslo_utils import timeutils
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
from keystone.oauth1 import core as oauth
from keystone.oauth1 import validator
from keystone.server import flask as ks_flask
Turn a signed request with an access key into a keystone token.