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
@validate(name=CAP_RTC)
def validate_cap_rtc(cap_rtc):
    if not isinstance(cap_rtc, bool):
        raise ConfigTypeError(desc='Invalid type for specifying RTC capability. Expected boolean got: %s' % type(cap_rtc))
    return cap_rtc