from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _assert_service_exists(self, service_id):
    try:
        if service_id is not None:
            self.get_service(service_id)
    except exception.ServiceNotFound:
        raise exception.ValidationError(attribute='endpoint service_id', target='service table')