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
def _show_software_deployment(self, cnxt, deployment_id):
    sd = software_deployment_object.SoftwareDeployment.get_by_id(cnxt, deployment_id)
    if sd.status == rpc_api.SOFTWARE_DEPLOYMENT_IN_PROGRESS:
        c = sd.config.config
        input_values = dict((swc_io.InputConfig(**i).input_data() for i in c[rpc_api.SOFTWARE_CONFIG_INPUTS]))
        transport = input_values.get('deploy_signal_transport')
        if transport == 'TEMP_URL_SIGNAL':
            sd = self._refresh_swift_software_deployment(cnxt, sd, input_values.get('deploy_signal_id'))
        elif transport == 'ZAQAR_SIGNAL':
            sd = self._refresh_zaqar_software_deployment(cnxt, sd, input_values.get('deploy_queue_id'))
    return sd