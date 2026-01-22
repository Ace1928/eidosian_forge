from keystoneclient import base
class CredentialsManager(base.ManagerWithFind):
    resource_class = EC2

    def create(self, user_id, tenant_id):
        """Create a new access/secret pair for the user/tenant pair.

        :rtype: object of type :class:`EC2`
        """
        params = {'tenant_id': tenant_id}
        return self._post('/users/%s/credentials/OS-EC2' % user_id, params, 'credential')

    def list(self, user_id):
        """Get a list of access/secret pairs for a user_id.

        :rtype: list of :class:`EC2`
        """
        return self._list('/users/%s/credentials/OS-EC2' % user_id, 'credentials')

    def get(self, user_id, access):
        """Get the access/secret pair for a given access key.

        :rtype: object of type :class:`EC2`
        """
        return self._get('/users/%s/credentials/OS-EC2/%s' % (user_id, base.getid(access)), 'credential')

    def delete(self, user_id, access):
        """Delete an access/secret pair for a user."""
        return self._delete('/users/%s/credentials/OS-EC2/%s' % (user_id, base.getid(access)))