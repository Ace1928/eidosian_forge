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
def _refresh_swift_software_deployment(self, cnxt, sd, deploy_signal_id):
    container, object_name = parse.urlparse(deploy_signal_id).path.split('/')[-2:]
    swift_plugin = cnxt.clients.client_plugin('swift')
    swift = swift_plugin.client()
    try:
        headers = swift.head_object(container, object_name)
    except Exception as ex:
        if swift_plugin.is_not_found(ex):
            LOG.info('Signal object not found: %(c)s %(o)s', {'c': container, 'o': object_name})
            return sd
        raise
    lm = headers.get('last-modified')
    last_modified = swift_plugin.parse_last_modified(lm)
    prev_last_modified = sd.updated_at
    if prev_last_modified:
        prev_last_modified = prev_last_modified.replace(tzinfo=None)
    if prev_last_modified and last_modified <= prev_last_modified:
        return sd
    try:
        headers, obj = swift.get_object(container, object_name)
    except Exception as ex:
        if swift_plugin.is_not_found(ex):
            LOG.info('Signal object not found: %(c)s %(o)s', {'c': container, 'o': object_name})
            return sd
        raise
    if obj:
        self.signal_software_deployment(cnxt, sd.id, jsonutils.loads(obj), last_modified.isoformat())
    return software_deployment_object.SoftwareDeployment.get_by_id(cnxt, sd.id)