from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def ParseIdentityConfigFile(dataproc, identity_config_file):
    """Parses a identity-config-file into the IdentityConfig message."""
    if identity_config_file.startswith('gs://'):
        data = storage_helpers.ReadObject(identity_config_file)
    else:
        data = console_io.ReadFromFileOrStdin(identity_config_file, binary=False)
    try:
        identity_config_data = yaml.load(data)
    except Exception as e:
        raise exceptions.ParseError('Cannot parse YAML:[{0}]'.format(e))
    user_service_account_mapping = encoding.DictToAdditionalPropertyMessage(identity_config_data.get('user_service_account_mapping', {}), dataproc.messages.IdentityConfig.UserServiceAccountMappingValue)
    identity_config_data_msg = dataproc.messages.IdentityConfig(userServiceAccountMapping=user_service_account_mapping)
    return identity_config_data_msg