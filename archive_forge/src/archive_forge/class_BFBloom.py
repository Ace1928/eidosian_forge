from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
class BFBloom(BFCommands, AbstractBloom):

    def __init__(self, client, **kwargs):
        """Create a new RedisBloom client."""
        _MODULE_CALLBACKS = {BF_RESERVE: bool_ok}
        _RESP2_MODULE_CALLBACKS = {BF_INFO: BFInfo}
        _RESP3_MODULE_CALLBACKS = {}
        self.client = client
        self.commandmixin = BFCommands
        self.execute_command = client.execute_command
        if get_protocol_version(self.client) in ['3', 3]:
            _MODULE_CALLBACKS.update(_RESP3_MODULE_CALLBACKS)
        else:
            _MODULE_CALLBACKS.update(_RESP2_MODULE_CALLBACKS)
        for k, v in _MODULE_CALLBACKS.items():
            self.client.set_response_callback(k, v)