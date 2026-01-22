from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def ConfigMapFlags(set_flag_only=False):
    return ResourceListFlagGroup(name='config-maps', set_flag_only=set_flag_only, help="Specify config maps to mount or provide as environment variables. Keys starting with a forward slash '/' are mount paths. All other keys correspond to environment variables. The values associated with each of these should be in the form CONFIG_MAP_NAME:KEY_IN_CONFIG_MAP; you may omit the key within the config map to specify a mount of all keys within the config map. For example: `--set-config-maps=/my/path=myconfig,ENV=otherconfig:key.json` will create a volume with config map 'myconfig' and mount that volume at '/my/path'. Because no config map key was specified, all keys in 'myconfig' will be included. An environment variable named ENV will also be created whose value is the value of 'key.json' in 'otherconfig'.")