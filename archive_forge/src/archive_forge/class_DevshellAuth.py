from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from boto.auth_handler import AuthHandler
from boto.auth_handler import NotReadyToAuthenticate
import oauth2client.contrib.devshell as devshell
class DevshellAuth(AuthHandler):
    """Developer Shell authorization plugin class."""
    capability = ['s3']

    def __init__(self, path, config, provider):
        if provider.name != 'google':
            raise NotReadyToAuthenticate()
        try:
            self.creds = devshell.DevshellCredentials()
        except:
            raise NotReadyToAuthenticate()

    def add_auth(self, http_request):
        http_request.headers['Authorization'] = 'Bearer %s' % self.creds.access_token