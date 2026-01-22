from oslo_log import log
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_endpoints_for_service(service_id, endpoints):
    return [ep for ep in endpoints if ep['service_id'] == service_id]