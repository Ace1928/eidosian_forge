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
class ConfWithIdListener(BaseConfListener):

    def __init__(self, conf_with_id):
        assert conf_with_id
        super(ConfWithIdListener, self).__init__(conf_with_id)
        conf_with_id.add_listener(ConfWithId.UPDATE_NAME_EVT, self.on_chg_name_conf_with_id)
        conf_with_id.add_listener(ConfWithId.UPDATE_DESCRIPTION_EVT, self.on_chg_desc_conf_with_id)

    def on_chg_name_conf_with_id(self, conf_evt):
        raise NotImplementedError()

    def on_chg_desc_conf_with_id(self, conf_evt):
        raise NotImplementedError()