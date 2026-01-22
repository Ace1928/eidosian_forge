from oslo_config import cfg
from oslo_utils import uuidutils
from glanceclient import client as gc
from glanceclient import exc
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def _findall_with_attr(self, entity, **kwargs):
    """Find all items for entity with attributes matching ``**kwargs``."""
    func = getattr(self.client(), entity)
    filters = {'filters': kwargs}
    return func.list(**filters)