from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class EndpointSpec(dict):
    """
    Describes properties to access and load-balance a service.

    Args:

        mode (string): The mode of resolution to use for internal load
          balancing between tasks (``'vip'`` or ``'dnsrr'``). Defaults to
          ``'vip'`` if not provided.
        ports (dict): Exposed ports that this service is accessible on from the
          outside, in the form of ``{ published_port: target_port }`` or
          ``{ published_port: <port_config_tuple> }``. Port config tuple format
          is ``(target_port [, protocol [, publish_mode]])``.
          Ports can only be provided if the ``vip`` resolution mode is used.
    """

    def __init__(self, mode=None, ports=None):
        if ports:
            self['Ports'] = convert_service_ports(ports)
        if mode:
            self['Mode'] = mode