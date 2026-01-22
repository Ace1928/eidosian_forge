from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.binauthz import parsing
from googlecloudsdk.core import log
def ResourceFileName(fname):
    if parsing.GetResourceFileType(fname) == parsing.ResourceFileType.UNKNOWN:
        raise arg_parsers.ArgumentTypeError('Resource file must be a .yaml or .json file.')
    return fname