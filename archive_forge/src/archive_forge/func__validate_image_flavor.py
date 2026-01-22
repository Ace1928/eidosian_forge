import copy
import ipaddress
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import port as neutron_port
from heat.engine.resources.openstack.neutron import subnet
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import server_base
from heat.engine import support
from heat.engine import translation
from heat.rpc import api as rpc_api
def _validate_image_flavor(self, image, flavor):
    try:
        image_obj = self.client_plugin('glance').get_image(image)
        flavor_obj = self.client_plugin().get_flavor(flavor)
    except Exception as ex:
        if self.client_plugin().is_not_found(ex) or self.client_plugin('glance').is_not_found(ex):
            return
        raise
    else:
        if image_obj.status.lower() != self.IMAGE_ACTIVE:
            msg = _('Image status is required to be %(cstatus)s not %(wstatus)s.') % {'cstatus': self.IMAGE_ACTIVE, 'wstatus': image_obj.status}
            raise exception.StackValidationFailed(message=msg)
        if flavor_obj.ram < image_obj.min_ram:
            msg = _('Image %(image)s requires %(imram)s minimum ram. Flavor %(flavor)s has only %(flram)s.') % {'image': image, 'imram': image_obj.min_ram, 'flavor': flavor, 'flram': flavor_obj.ram}
            raise exception.StackValidationFailed(message=msg)
        if flavor_obj.disk < image_obj.min_disk:
            msg = _('Image %(image)s requires %(imsz)s GB minimum disk space. Flavor %(flavor)s has only %(flsz)s GB.') % {'image': image, 'imsz': image_obj.min_disk, 'flavor': flavor, 'flsz': flavor_obj.disk}
            raise exception.StackValidationFailed(message=msg)