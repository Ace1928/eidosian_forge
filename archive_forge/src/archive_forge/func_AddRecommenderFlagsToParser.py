from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def AddRecommenderFlagsToParser(parser, entities):
    """Adds argument mutex group of specified entities and recommender to parser.

  Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
      entities: The entities to add.
  """
    AddEntityFlagsToParser(parser, entities)
    parser.add_argument('--location', metavar='LOCATION', required=True, help='Location to use for this invocation.')
    parser.add_argument('recommender', metavar='RECOMMENDER', help='Recommender to use for this invocation.')