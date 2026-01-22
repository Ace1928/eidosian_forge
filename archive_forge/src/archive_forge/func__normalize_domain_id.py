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
@classmethod
def _normalize_domain_id(cls, ref):
    """Fill in domain_id if not specified in a v3 call."""
    if not ref.get('domain_id'):
        oslo_ctx = flask.request.environ.get(context.REQUEST_CONTEXT_ENV, None)
        if oslo_ctx and oslo_ctx.domain_id:
            ref['domain_id'] = oslo_ctx.domain_id
        elif oslo_ctx.is_admin:
            raise exception.ValidationError(_('You have tried to create a resource using the admin token. As this token is not within a domain you must explicitly include a domain for this resource to belong to.'))
        else:
            versionutils.report_deprecated_feature(LOG, 'Not specifying a domain during a create user, group or project call, and relying on falling back to the default domain, is deprecated as of Liberty. There is no plan to remove this compatibility, however, future API versions may remove this, so please specify the domain explicitly or use a domain-scoped token.')
            ref['domain_id'] = CONF.identity.default_domain_id
    return ref