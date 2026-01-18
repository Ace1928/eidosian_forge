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
@validate(name=MAX_PREFIXES)
def validate_max_prefixes(max_prefixes):
    if not isinstance(max_prefixes, int):
        raise ConfigTypeError(desc='Max. prefixes value should be of type int or long but found %s' % type(max_prefixes))
    if max_prefixes < 0:
        raise ConfigValueError(desc='Invalid max. prefixes value: %s' % max_prefixes)
    return max_prefixes