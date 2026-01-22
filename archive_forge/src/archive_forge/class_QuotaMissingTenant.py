from oslo_utils import excutils
from neutron_lib._i18n import _
class QuotaMissingTenant(BadRequest):
    message = _('Tenant-id was missing from quota request.')