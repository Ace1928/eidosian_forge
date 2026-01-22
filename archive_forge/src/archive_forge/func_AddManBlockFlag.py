from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddManBlockFlag(parser):
    """Adds --man-block flag."""
    parser.add_argument('--man-block', help="Master Authorized Network. Allows access to the Kubernetes control plane from this block. Defaults to '0.0.0.0/0' if flag is not provided.")