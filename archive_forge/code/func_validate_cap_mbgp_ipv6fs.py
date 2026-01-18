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
@validate(name=CAP_MBGP_IPV6FS)
def validate_cap_mbgp_ipv6fs(cmv6fs):
    if not isinstance(cmv6fs, bool):
        raise ConfigTypeError(desc='Invalid MP-BGP IPv6 Flow Specification capability settings: %s. Boolean value expected' % cmv6fs)
    return cmv6fs