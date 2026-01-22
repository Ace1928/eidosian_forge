from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
class CFBloom(CFCommands, AbstractBloom):

    def __init__(self, client, **kwargs):
        """Create a new RedisBloom client."""
        _MODULE_CALLBACKS = {CF_RESERVE: bool_ok}
        _RESP2_MODULE_CALLBACKS = {CF_INFO: CFInfo}
        _RESP3_MODULE_CALLBACKS = {}
        self.client = client
        self.commandmixin = CFCommands
        self.execute_command = client.execute_command
        if get_protocol_version(self.client) in ['3', 3]:
            _MODULE_CALLBACKS.update(_RESP3_MODULE_CALLBACKS)
        else:
            _MODULE_CALLBACKS.update(_RESP2_MODULE_CALLBACKS)
        for k, v in _MODULE_CALLBACKS.items():
            self.client.set_response_callback(k, v)