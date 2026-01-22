from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import names
import six
def _AddToArgparse(self, attributes, parser):
    """Recursively add an arg definition to the parser."""
    for attribute in attributes:
        if isinstance(attribute, base.Attribute):
            parser.add_argument(attribute.arg_name, **attribute.kwargs)
            continue
        group = parser.add_argument_group(attribute.kwargs.pop('help'), **attribute.kwargs)
        self._AddToArgparse(attribute.attributes, group)