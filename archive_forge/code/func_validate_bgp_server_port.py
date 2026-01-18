import logging
import numbers
from os_ken.lib import ip
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
from os_ken.services.protocols.bgp import rtconf
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import compute_optional_conf
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import MissingRequiredConf
from os_ken.services.protocols.bgp.rtconf.base import validate
@validate(name=BGP_SERVER_PORT)
def validate_bgp_server_port(server_port):
    if not isinstance(server_port, numbers.Integral):
        raise ConfigTypeError(desc='Invalid bgp sever port configuration value %s' % server_port)
    if server_port < 0 or server_port > 65535:
        raise ConfigValueError(desc='Invalid server port %s' % server_port)
    return server_port