from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def update_volume_target(self, volume_target, **attrs):
    """Update a volume_target.

        :param volume_target: Either the ID of a volume_target
            or an instance of
            :class:`~openstack.baremetal.v1.volume_target.VolumeTarget`.
        :param dict attrs: The attributes to update on the
            volume_target represented by the ``volume_target`` parameter.

        :returns: The updated volume_target.
        :rtype:
            :class:`~openstack.baremetal.v1.volume_target.VolumeTarget`
        """
    return self._update(_volumetarget.VolumeTarget, volume_target, **attrs)