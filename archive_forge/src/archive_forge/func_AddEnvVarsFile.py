from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def AddEnvVarsFile(self):
    self._AddFlag('--env-vars-file', metavar='FILE_PATH', type=map_util.ArgDictFile(key_type=six.text_type, value_type=six.text_type), help='Path to a local YAML file with definitions for all environment variables.')