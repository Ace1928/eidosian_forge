from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import walker_util
def ConvertPathToIdentifier(path):
    return _COMPLETIONS_PREFIX + '__'.join(path).replace('-', '_')