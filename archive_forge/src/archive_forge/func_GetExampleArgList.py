from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def GetExampleArgList(self):
    """Returns a list of command line example arg strings for the concept."""
    args = self.GetAttributeArgs()
    examples = []
    for arg in args:
        if arg.name.startswith('--'):
            example = '{}=my-{}'.format(arg.name, arg.name[2:])
        else:
            example = 'my-{}'.format(arg.name.lower())
        examples.append(example)
    return examples