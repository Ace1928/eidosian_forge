import abc
from abc import ABCMeta
from abc import abstractmethod
import functools
import numbers
import logging
import uuid
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.base import RUNTIME_CONF_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
@validate(name=CAP_MBGP_VPNV4)
def validate_cap_mbgp_vpnv4(cmv4):
    if not isinstance(cmv4, bool):
        raise ConfigTypeError(desc='Invalid MP-BGP VPNv4 capability settings: %s. Boolean value expected' % cmv4)
    return cmv4