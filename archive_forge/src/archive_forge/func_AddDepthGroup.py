from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
def AddDepthGroup(parser):
    """Adds necessary depth flags for discover command parser."""
    depth_parser = parser.add_group(mutex=True)
    depth_parser.add_argument('--recursive', help='Whether to retrieve the full hierarchy of data objects (TRUE) or only the current level (FALSE).', action=actions.DeprecationAction('--recursive', warn='The {flag_name} option is deprecated; use `--full-hierarchy` instead.', removed=False, action='store_true'))
    depth_parser.add_argument('--recursive-depth', help='The number of hierarchy levels below the current level to be retrieved.', action=actions.DeprecationAction('--recursive-depth', warn='The {flag_name} option is deprecated; use `--hierarchy-depth` instead.', removed=False))