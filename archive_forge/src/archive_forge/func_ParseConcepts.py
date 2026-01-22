from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import names
import six
def ParseConcepts(self):
    """Parse all concepts.

    Stores the result of parsing concepts, keyed to the namespace format of
    their presentation name. Afterward, will be accessible as
    args.<LOWER_SNAKE_CASE_NAME>.

    Raises:
      googlecloudsdk.command_lib.concepts.exceptions.Error: if parsing fails.
    """
    final = {}
    for attr_name, attribute in six.iteritems(self._attributes):
        dependencies = dependency_managers.DependencyNode.FromAttribute(attribute)
        final[attr_name] = FinalParse(dependencies, self.ParsedArgs)
    for name, value in six.iteritems(final):
        setattr(self.parsed_args, name, value)