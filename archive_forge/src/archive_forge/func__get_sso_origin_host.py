import string
import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import strutils
import urllib
import werkzeug.exceptions
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.api._shared import saml
from keystone.auth import schema as auth_schema
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import utils as k_utils
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.federation import schema as federation_schema
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _get_sso_origin_host():
    """Validate and return originating dashboard URL.

    Make sure the parameter is specified in the request's URL as well its
    value belongs to a list of trusted dashboards.

    :raises keystone.exception.ValidationError: ``origin`` query parameter
        was not specified. The URL is deemed invalid.
    :raises keystone.exception.Unauthorized: URL specified in origin query
        parameter does not exist in list of websso trusted dashboards.
    :returns: URL with the originating dashboard

    """
    origin = flask.request.args.get('origin')
    if not origin:
        msg = 'Request must have an origin query parameter'
        tr_msg = _('Request must have an origin query parameter')
        LOG.error(msg)
        raise exception.ValidationError(tr_msg)
    host = urllib.parse.unquote_plus(origin)
    trusted_dashboards = [k_utils.lower_case_hostname(trusted) for trusted in CONF.federation.trusted_dashboard]
    if host not in trusted_dashboards:
        msg = '%(host)s is not a trusted dashboard host' % {'host': host}
        tr_msg = _('%(host)s is not a trusted dashboard host') % {'host': host}
        LOG.error(msg)
        raise exception.Unauthorized(tr_msg)
    return host