from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def AddAnnotationsToParser(parser, resource):
    """Adds annotations to parser.

  Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
      resource: The resource to add to.
  """
    parser.add_argument('--annotations', type=arg_parsers.ArgDict(min_length=1), default={}, help='Store small amounts of arbitrary data on the {}.'.format(resource), metavar='KEY=VALUE', action=arg_parsers.StoreOnceAction)