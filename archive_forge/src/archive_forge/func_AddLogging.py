from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddLogging(parser, allow_disabled=False):
    """Adds the --logging flag."""
    help_text = '\nSet the components that have logging enabled.\n\nExamples:\n\n  $ {command} --logging=SYSTEM\n  $ {command} --logging=SYSTEM,WORKLOAD'
    logging_choices = []
    if allow_disabled:
        logging_choices = _ALLOW_DISABLE_LOGGING_CHOICES
        help_text += '\n  $ {command} --logging=NONE\n'
    else:
        logging_choices = _LOGGING_CHOICES
    parser.add_argument('--logging', type=arg_parsers.ArgList(min_length=1, choices=logging_choices), metavar='COMPONENT', help=help_text)