from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
class CoverageTreeGenerator(walker.Walker):
    """Generates the gcloud static completion CLI tree."""

    def __init__(self, cli=None, branch=None, restrict=None):
        """branch is the command path of the CLI subtree to generate."""
        super(CoverageTreeGenerator, self).__init__(cli=cli, restrict=restrict)
        self._branch = branch

    def Visit(self, node, parent, is_group):
        """Visits each node in the CLI command tree to construct the external rep.

    Args:
      node: group/command CommandCommon info.
      parent: The parent Visit() return value, None at the top level.
      is_group: True if node is a command group.

    Returns:
      The subtree parent value, used here to construct an external rep node.
    """
        return CoverageCommandNode(node, parent)