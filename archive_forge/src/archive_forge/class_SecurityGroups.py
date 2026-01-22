from troveclient import base
from troveclient import common
class SecurityGroups(base.ManagerWithFind):
    """Manage :class:`SecurityGroup` resources."""
    resource_class = SecurityGroup

    def list(self, limit=None, marker=None):
        """Get a list of all security groups.

        :rtype: list of :class:`SecurityGroup`.
        """
        return self._paginated('/security-groups', 'security_groups', limit, marker)

    def get(self, security_group):
        """Get a specific security group.

        :rtype: :class:`SecurityGroup`
        """
        return self._get('/security-groups/%s' % base.getid(security_group), 'security_group')