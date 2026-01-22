from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def _call_determining_protocol_version(self):
    """Determine what protocol the remote server supports.

        We do this by placing a request in the most recent protocol, and
        handling the UnexpectedProtocolVersionMarker from the server.
        """
    last_err = None
    for protocol_version in [3, 2]:
        if protocol_version == 2:
            self.client._medium._remember_remote_is_before((1, 6))
        try:
            response_tuple, response_handler = self._call(protocol_version)
        except errors.UnexpectedProtocolVersionMarker as err:
            trace.warning('Server does not understand Bazaar network protocol %d, reconnecting.  (Upgrade the server to avoid this.)' % (protocol_version,))
            self.client._medium.disconnect()
            last_err = err
            continue
        except errors.ErrorFromSmartServer:
            self.client._medium._protocol_version = protocol_version
            raise
        else:
            self.client._medium._protocol_version = protocol_version
            return (response_tuple, response_handler)
    raise errors.SmartProtocolError('Server is not a Bazaar server: ' + str(last_err))