import itertools
import uuid
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import requests
from urllib import parse
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.engine import api
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.rpc import api as rpc_api
def _refresh_zaqar_software_deployment(self, cnxt, sd, deploy_queue_id):
    rs = db_api.resource_get_by_physical_resource_id(cnxt, sd.id)
    project = sd.stack_user_project_id
    queue = self._get_zaqar_queue(cnxt, rs, project, deploy_queue_id)
    messages = list(queue.pop())
    if messages:
        self.signal_software_deployment(cnxt, sd.id, messages[0].body, None)
    return software_deployment_object.SoftwareDeployment.get_by_id(cnxt, sd.id)