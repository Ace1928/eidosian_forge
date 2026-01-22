from oslo_config import cfg
from webob import exc
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
Validate auth_url to ensure it can be used.