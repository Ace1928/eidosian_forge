import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.command_lib.scc.manage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
def GetCustomConfigFromArgs(file):
    """Process the custom config file for the custom module."""
    if file is not None:
        try:
            config_dict = yaml.load(file)
            return encoding.DictToMessage(config_dict, messages.CustomConfig)
        except yaml.YAMLParseError as ype:
            raise errors.InvalidCustomConfigFileError('Error parsing custom config file [{}]'.format(ype))