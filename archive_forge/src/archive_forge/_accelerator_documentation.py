from openstack.accelerator.v2._proxy import Proxy
Unbind an accelerator from VM.

        :param uuid: The uuid of the accelerator_request to be unbinded.
        :param properties: The info of VM that will unbind the accelerator.
        :returns: True if unbind succeeded, False otherwise.
        