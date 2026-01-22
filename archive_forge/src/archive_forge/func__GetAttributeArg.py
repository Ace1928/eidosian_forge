from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def _GetAttributeArg(self, attribute):
    """Creates argument for a specific attribute."""
    name = self.attribute_to_args_map.get(attribute.name, None)
    if not name:
        return None
    return base.Argument(name, **self._KwargsForAttribute(name, attribute))