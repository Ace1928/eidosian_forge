from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def update_accelerator_request(self, uuid, properties):
    """Bind/Unbind an accelerator to VM.

        :param uuid: The uuid of the accelerator_request to be bound/unbound.
        :param properties: The info of VM
            that will bind/unbind the accelerator.
        :returns: True if bind/unbind succeeded, False otherwise.
        """
    return self._get_resource(_arq.AcceleratorRequest, uuid).patch(self, properties)