from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class _ProjectNumberArgFallthrough(deps.ArgFallthrough):
    """A fallthrough for project number from the --project argument."""

    def __init__(self):
        """See base class."""
        super(_ProjectNumberArgFallthrough, self).__init__(arg_name='--project')

    def _Call(self, parsed_args):
        """See base class."""
        return _EnsureProjectNumber(super(_ProjectNumberArgFallthrough, self)._Call(parsed_args))