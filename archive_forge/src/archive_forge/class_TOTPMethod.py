import copy
from keystoneauth1.identity.v3 import base
class TOTPMethod(base.AuthMethod):
    """Construct a User/Passcode based authentication method.

    :param string passcode: TOTP passcode for authentication.
    :param string username: Username for authentication.
    :param string user_id: User ID for authentication.
    :param string user_domain_id: User's domain ID for authentication.
    :param string user_domain_name: User's domain name for authentication.
    """
    _method_parameters = ['user_id', 'username', 'user_domain_id', 'user_domain_name', 'passcode']

    def get_auth_data(self, session, auth, headers, **kwargs):
        user = {'passcode': self.passcode}
        if self.user_id:
            user['id'] = self.user_id
        elif self.username:
            user['name'] = self.username
            if self.user_domain_id:
                user['domain'] = {'id': self.user_domain_id}
            elif self.user_domain_name:
                user['domain'] = {'name': self.user_domain_name}
        return ('totp', {'user': user})

    def get_cache_id_elements(self):
        params = copy.copy(self._method_parameters)
        params.remove('passcode')
        return dict((('totp_%s' % p, getattr(self, p)) for p in self._method_parameters))