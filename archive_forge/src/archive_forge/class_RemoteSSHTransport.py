imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
class RemoteSSHTransport(RemoteTransport):
    """Connection to smart server over SSH.

    This is essentially just a factory to get 'RemoteTransport(url,
        SmartSSHClientMedium).
    """

    def _build_medium(self):
        location_config = config.LocationConfig(self.base)
        bzr_remote_path = location_config.get_bzr_remote_path()
        user = self._parsed_url.user
        if user is None:
            auth = config.AuthenticationConfig()
            user = auth.get_user('ssh', self._parsed_url.host, self._parsed_url.port)
        ssh_params = medium.SSHParams(self._parsed_url.host, self._parsed_url.port, user, self._parsed_url.password, bzr_remote_path)
        client_medium = medium.SmartSSHClientMedium(self.base, ssh_params)
        return (client_medium, (user, self._parsed_url.password))