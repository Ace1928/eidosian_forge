imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
class RemoteTCPTransportV2Only(RemoteTransport):
    """Connection to smart server over plain tcp with the client hard-coded to
    assume protocol v2 and remote server version <= 1.6.

    This should only be used for testing.
    """

    def _build_medium(self):
        client_medium = medium.SmartTCPClientMedium(self._parsed_url.host, self._parsed_url.port, self.base)
        client_medium._protocol_version = 2
        client_medium._remember_remote_is_before((1, 6))
        return (client_medium, None)