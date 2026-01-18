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
@validate(name=TCP_CONN_TIMEOUT)
def validate_tcp_conn_timeout(tcp_conn_timeout):
    if not isinstance(tcp_conn_timeout, numbers.Integral):
        raise ConfigTypeError(desc='Invalid tcp connection timeout configuration value %s' % tcp_conn_timeout)
    if tcp_conn_timeout < 10:
        raise ConfigValueError(desc='Invalid tcp connection timeout configuration value %s' % tcp_conn_timeout)
    return tcp_conn_timeout