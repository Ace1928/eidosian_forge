import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def validate_expiration(token):
    token_expiration_datetime = timeutils.normalize_time(timeutils.parse_isotime(token.expires_at))
    if timeutils.utcnow() > token_expiration_datetime:
        raise exception.Unauthorized(_('Federation token is expired'))