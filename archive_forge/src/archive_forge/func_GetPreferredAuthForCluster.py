from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import copy
import json
import os
from googlecloudsdk.command_lib.anthos import flags
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
from six.moves import urllib
def GetPreferredAuthForCluster(cluster, login_config, config_contents=None, force_update=False, is_url=False):
    """Get preferredAuthentication value for cluster."""
    if not (cluster and login_config):
        return (None, None, None)
    configs = None
    if is_url:
        if not config_contents:
            raise AnthosAuthException('Config contents were not passed with URL [{}]'.format(login_config))
        configs = file_parsers.YamlConfigFile(file_contents=config_contents, item_type=file_parsers.LoginConfigObject)
    else:
        configs = file_parsers.YamlConfigFile(file_contents=config_contents, file_path=login_config, item_type=file_parsers.LoginConfigObject)
    cluster_config = _GetClusterConfig(configs, cluster)
    try:
        auth_method = cluster_config.GetPreferredAuth()
    except KeyError:
        auth_method = None
    except file_parsers.YamlConfigObjectFieldError:
        return (None, None, None)
    if not auth_method or force_update:
        providers = cluster_config.GetAuthProviders()
        if not providers:
            raise AnthosAuthException('No Authentication Providers found in [{}]'.format(login_config))
        if len(providers) == 1:
            auth_method = providers.pop()
        else:
            prompt_message = 'Please select your preferred authentication option for cluster [{}]'.format(cluster)
            override_warning = '. Note: This will overwrite current preferred auth method [{}] in config file.'
            if auth_method and force_update and (not is_url):
                prompt_message = prompt_message + override_warning.format(auth_method)
            index = console_io.PromptChoice(providers, message=prompt_message, cancel_option=True)
            auth_method = providers[index]
        log.status.Print('Setting Preferred Authentication option to [{}]'.format(auth_method))
        cluster_config.SetPreferredAuth(auth_method)
        if login_config and (not is_url):
            configs.WriteToDisk()
    ldap_user, ldap_pass = _GetLdapUserAndPass(cluster_config, auth_method, cluster)
    return (auth_method, ldap_user, ldap_pass)