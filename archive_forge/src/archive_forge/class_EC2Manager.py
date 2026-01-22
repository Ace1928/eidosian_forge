from keystoneclient import base
class EC2Manager(base.ManagerWithFind):
    resource_class = EC2

    def create(self, user_id, project_id):
        """Create a new access/secret pair.

        :param user_id: the ID of the user having access/secret pair.
        :type user_id: str or :class:`keystoneclient.v3.users.User`
        :param project_id: the ID of the project having access/secret pair.
        :type project_id: str or :class:`keystoneclient.v3.projects.Project`

        :returns: the created access/secret pair returned from server.
        :rtype: :class:`keystoneclient.v3.ec2.EC2`

        """
        return self._post('/users/%s/credentials/OS-EC2' % user_id, body={'tenant_id': project_id}, response_key='credential')

    def get(self, user_id, access):
        """Retrieve an access/secret pair for a given access key.

        :param user_id: the ID of the user whose access/secret pair will be
                        retrieved from the server.
        :type user_id: str or :class:`keystoneclient.v3.users.User`
        :param str access: the access key whose access/secret pair will be
                           retrieved from the server.

        :returns: the specified access/secret pair returned from server.
        :rtype: :class:`keystoneclient.v3.ec2.EC2`

        """
        url = '/users/%s/credentials/OS-EC2/%s' % (user_id, base.getid(access))
        return self._get(url, response_key='credential')

    def list(self, user_id):
        """List access/secret pairs for a given user.

        :param str user_id: the ID of the user having access/secret pairs will
                            be listed.

        :returns: a list of access/secret pairs.
        :rtype: list of :class:`keystoneclient.v3.ec2.EC2`

        """
        return self._list('/users/%s/credentials/OS-EC2' % user_id, response_key='credentials')

    def delete(self, user_id, access):
        """Delete an access/secret pair.

        :param user_id: the ID of the user whose access/secret pair will be
                        deleted on the server.
        :type user_id: str or :class:`keystoneclient.v3.users.User`
        :param str access: the access key whose access/secret pair will be
                           deleted on the server.

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        """
        return self._delete('/users/%s/credentials/OS-EC2/%s' % (user_id, base.getid(access)))