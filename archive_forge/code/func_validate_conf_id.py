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
@validate(name=ConfWithId.ID)
def validate_conf_id(identifier):
    if not isinstance(identifier, str):
        raise ConfigTypeError(conf_name=ConfWithId.ID, conf_value=identifier)
    if len(identifier) > 128:
        raise ConfigValueError(conf_name=ConfWithId.ID, conf_value=identifier)
    return identifier