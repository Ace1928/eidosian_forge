from oslo_log import log as logging
from oslo_serialization import jsonutils
import tempfile
from heat.common import auth_plugin
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import template
def _prepare_region_context(self):
    dict_ctxt = self.context.to_dict()
    dict_ctxt.update({'region_name': self._region_name, 'overwrite': False})
    self._local_context = context.RequestContext.from_dict(dict_ctxt)
    if self._ssl_verify is not None:
        self._local_context.keystone_session.verify = self._ssl_verify
    return self._local_context