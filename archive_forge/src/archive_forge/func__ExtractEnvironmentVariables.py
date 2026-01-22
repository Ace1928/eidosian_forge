from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
from googlecloudsdk.core.util import encoding
def _ExtractEnvironmentVariables(self):
    """ExtractEnvironmentVariables can be used to extract environment variables required for binary operations.
    """
    self.env_vars = {'GOOGLE_OAUTH_ACCESS_TOKEN': GetFreshAccessToken(account=properties.VALUES.core.account.Get()), 'USE_STRUCTURED_LOGGING': 'true'}
    proxy_env_names = ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy']
    project_env_names = ['GOOGLE_PROJECT', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT']
    for env_key, env_val in os.environ.items():
        if env_key in proxy_env_names:
            self.env_vars[env_key] = env_val
    self.project = properties.VALUES.core.project.Get()
    if self.project:
        log.debug('Setting project to {} from properties'.format(self.project))
    else:
        for env_key in project_env_names:
            self.project = encoding.GetEncodedValue(os.environ, env_key)
            if self.project:
                log.debug('Setting project to {} from env {}'.format(self.project, env_key))
                break
        if not self.project:
            raise exceptions.Error('Failed to retrieve the project id. Please specify the project id using --project flag.')