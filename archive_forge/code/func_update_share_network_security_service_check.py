from manilaclient import api_versions
from manilaclient import base
from manilaclient import exceptions
@api_versions.wraps('2.63')
def update_share_network_security_service_check(self, share_network, current_security_service, new_security_service, reset_operation=False):
    """Validates if the security service update is supported by all hosts.

        :param share_network: share network name, id or ShareNetwork instance
        :param current_security_service: current name, id or
        SecurityService instance that will be changed
        :param new_security_service: new name, id or
        :param reset_operation: start over the check operation
        SecurityService instance that will be updated
        :rtype: :class:`ShareNetwork`
        """
    info = {'current_service_id': base.getid(current_security_service), 'new_service_id': base.getid(new_security_service), 'reset_operation': reset_operation}
    return self._action('update_security_service_check', share_network, info)