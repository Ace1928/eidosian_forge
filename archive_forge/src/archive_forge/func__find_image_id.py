from oslo_config import cfg
from oslo_utils import uuidutils
from glanceclient import client as gc
from glanceclient import exc
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
@os_client.MEMOIZE_FINDER
def _find_image_id(self, tenant_id, image_identifier):
    return self.get_image(image_identifier).id