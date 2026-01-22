import os
from keystoneauth1 import loading
from keystoneauth1 import plugin
class AodhNoAuthPlugin(plugin.BaseAuthPlugin):
    """No authentication plugin for Aodh

    This is a keystoneauth plugin that instead of
    doing authentication, it just fill the 'x-user-id'
    and 'x-project-id' headers with the user provided one.
    """

    def __init__(self, user_id, project_id, roles, endpoint):
        self._user_id = user_id
        self._project_id = project_id
        self._endpoint = endpoint
        self._roles = roles

    def get_token(self, session, **kwargs):
        return '<no-token-needed>'

    def get_headers(self, session, **kwargs):
        return {'x-user-id': self._user_id, 'x-project-id': self._project_id, 'x-roles': self._roles}

    def get_user_id(self, session, **kwargs):
        return self._user_id

    def get_project_id(self, session, **kwargs):
        return self._project_id

    def get_endpoint(self, session, **kwargs):
        return self._endpoint