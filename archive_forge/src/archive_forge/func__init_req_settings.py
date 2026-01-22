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
def _init_req_settings(self, **kwargs):
    for req_attr in self._req_settings:
        req_attr_value = kwargs.get(req_attr)
        if req_attr_value is None:
            raise MissingRequiredConf(conf_name=req_attr_value)
        req_attr_value = get_validator(req_attr)(req_attr_value)
        self._settings[req_attr] = req_attr_value