from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
class SmartClientHooks(hooks.Hooks):

    def __init__(self):
        hooks.Hooks.__init__(self, 'breezy.bzr.smart.client', '_SmartClient.hooks')
        self.add_hook('call', 'Called when the smart client is submitting a request to the smart server. Called with a breezy.bzr.smart.client.CallHookParams object. Streaming request bodies, and responses, are not accessible.', None)