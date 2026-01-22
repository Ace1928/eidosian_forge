from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as core_resources
import six
def PatchDryRunConfig(self, perimeter_ref, description=None, title=None, perimeter_type=None, resources=None, restricted_services=None, levels=None, vpc_allowed_services=None, enable_vpc_accessible_services=None, ingress_policies=None, egress_policies=None):
    """Patch the dry-run config (spec) for a Service Perimeter.

    Args:
      perimeter_ref: resources.Resource, reference to the perimeter to patch
      description: str, description of the zone or None if not updating
      title: str, title of the zone or None if not updating
      perimeter_type: PerimeterTypeValueValuesEnum type enum value for the level
        or None if not updating
      resources: list of str, the names of resources (for now, just
        'projects/...') in the zone or None if not updating.
      restricted_services: list of str, the names of services
        ('example.googleapis.com') that *are* restricted by the access zone or
        None if not updating.
      levels: list of Resource, the access levels (in the same policy) that must
        be satisfied for calls into this zone or None if not updating.
      vpc_allowed_services: list of str, the names of services
        ('example.googleapis.com') that *are* allowed to be made within the
        access zone, or None if not updating.
      enable_vpc_accessible_services: bool, whether to restrict the set of APIs
        callable within the access zone, or None if not updating.
      ingress_policies: list of IngressPolicy, or None if not updating.
      egress_policies: list of EgressPolicy, or None if not updating.

    Returns:
      ServicePerimeter, the updated Service Perimeter.
    """
    m = self.messages
    perimeter = m.ServicePerimeter()
    update_mask = []
    if _SetIfNotNone('title', title, perimeter, update_mask):
        perimeter.name = perimeter_ref.RelativeName()
        update_mask.append('name')
    _SetIfNotNone('description', description, perimeter, update_mask)
    _SetIfNotNone('perimeterType', perimeter_type, perimeter, update_mask)
    config, config_mask_additions = _CreateServicePerimeterConfig(messages=m, mask_prefix='spec', resources=resources, restricted_services=restricted_services, levels=levels, vpc_allowed_services=vpc_allowed_services, enable_vpc_accessible_services=enable_vpc_accessible_services, ingress_policies=ingress_policies, egress_policies=egress_policies)
    perimeter.spec = config
    update_mask.extend(config_mask_additions)
    perimeter.useExplicitDryRunSpec = True
    update_mask.append('useExplicitDryRunSpec')
    return self._ApplyPatch(perimeter_ref, perimeter, update_mask)