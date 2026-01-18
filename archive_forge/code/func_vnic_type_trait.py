import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def vnic_type_trait(vnic_type):
    """A Placement trait name to represent support for a vnic_type.

    :param physnet: The vnic_type.
    :returns: The trait name representing the vnic_type.
    """
    return os_traits.normalize_name('%s%s' % (place_const.TRAIT_PREFIX_VNIC_TYPE, vnic_type))