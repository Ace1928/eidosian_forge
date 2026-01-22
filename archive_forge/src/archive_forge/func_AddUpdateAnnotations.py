from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddUpdateAnnotations(parser: parser_arguments.ArgumentInterceptor):
    """Adds flags to update annotations.

  Args:
    parser: The argparse parser to add the flag to.
  """
    annotations_mutex_group = parser.add_group(mutex=True)
    annotations_mutex_group.add_argument('--add-annotations', metavar='KEY1=VALUE1,KEY2=VALUE2', help='Add the given key-value pairs to the current annotations, or update its value if the key already exists.', type=arg_parsers.ArgDict())
    annotations_mutex_group.add_argument('--remove-annotations', metavar='KEY1,KEY2', help='Remove annotations of the given keys.', type=arg_parsers.ArgList())
    annotations_mutex_group.add_argument('--clear-annotations', hidden=True, action='store_true', help='Clear all the current annotations')
    annotations_mutex_group.add_argument('--set-annotations', hidden=True, metavar='KEY1=VALUE1,KEY2=VALUE2', type=arg_parsers.ArgDict(), help='Replace all the current annotations')