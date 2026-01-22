from oslo_log import log as logging
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _deleter(self, obj=None):
    """Delete the underlying container or an object inside it."""
    args = [self.resource_id]
    if obj:
        deleter = self.client().delete_object
        args.append(obj['name'])
    else:
        deleter = self.client().delete_container
    with self.client_plugin().ignore_not_found:
        deleter(*args)