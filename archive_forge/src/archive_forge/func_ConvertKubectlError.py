from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import time
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def ConvertKubectlError(self, error, env_obj):
    is_private = env_obj.config.privateEnvironmentConfig and env_obj.config.privateEnvironmentConfig.enablePrivateEnvironment
    if is_private:
        return command_util.Error(six.text_type(error) + ' Make sure you have followed https://cloud.google.com/composer/docs/how-to/accessing/airflow-cli#running_commands_on_a_private_ip_environment to enable access to your private Cloud Composer environment from your machine.')
    return error