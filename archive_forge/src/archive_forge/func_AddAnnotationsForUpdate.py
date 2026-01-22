from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddAnnotationsForUpdate(parser, noun):
    """Adds annotations related flags for update.

  Args:
    parser: The argparse.parser to add the arguments to.
    noun: The resource type to which the flag is applicable.
  """
    group = parser.add_group('Annotations', mutex=True)
    AddAnnotations(group, noun)
    AddClearAnnotations(group, noun)