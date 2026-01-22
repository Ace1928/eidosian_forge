from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def MapUpdateFlag(flag_name, long_name, key_type, value_type, key_metavar='KEY', value_metavar='VALUE'):
    return base.Argument('--update-{}'.format(flag_name), metavar='{}={}'.format(key_metavar, value_metavar), action=arg_parsers.UpdateAction, type=arg_parsers.ArgDict(key_type=key_type, value_type=value_type), help='List of key-value pairs to set as {}.'.format(long_name))