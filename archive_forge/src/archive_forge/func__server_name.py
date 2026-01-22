import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def _server_name(self):
    name = self.properties[self.NAME]
    if name:
        return name
    return self.physical_resource_name()