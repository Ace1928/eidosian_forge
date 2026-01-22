from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def AddMapSetFileFlag(group, flag_name, long_name, key_type, value_type):
    group.add_argument('--{}-file'.format(flag_name), metavar='FILE_PATH', type=ArgDictFile(key_type=key_type, value_type=value_type), help='Path to a local YAML file with definitions for all {0}. All existing {0} will be removed before the new {0} are added.'.format(long_name))