from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTagValueArgToParser(parser):
    """Adds the TagValue argument to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('--tag-value', metavar='TAG_VALUE', required=True, help='Tag value name or namespaced name. The name should be in the form tagValues/{numeric_id}. The namespaced name should be in the form {org_id}/{tag_key_short_name}/{short_name} where short_name must be 1-63 characters, beginning and ending with an alphanumeric character ([a-z0-9A-Z]) with dashes (-), underscores (_), dots (.), and alphanumerics between.')