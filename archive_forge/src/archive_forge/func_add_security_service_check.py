from manilaclient import api_versions
from manilaclient import base
from manilaclient import exceptions
@api_versions.wraps('2.63')
def add_security_service_check(self, share_network, security_service, reset_operation=False):
    """Associate given security service with a share network.

        :param share_network: share network name, id or ShareNetwork instance
        :param security_service: name, id or SecurityService instance
        :param reset_operation: start over the check operation
        :rtype: :class:`ShareNetwork`
        """
    info = {'security_service_id': base.getid(security_service), 'reset_operation': reset_operation}
    return self._action('add_security_service_check', share_network, info)