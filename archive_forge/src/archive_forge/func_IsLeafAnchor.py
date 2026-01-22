from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def IsLeafAnchor(self, attribute):
    """Returns True if attribute is an anchor in at least one concept.

    Attribute can only be a leaf anchor if it is an anchor for at least
    one concept AND not an attribute in any other resource.

    Args:
      attribute: concepts.Attribute, attribute we are checking

    Returns:
      bool, whether attribute is a leaf anchor
    """
    if not self.IsAnchor(attribute):
        return False
    if any((attribute in spec.attributes and attribute.name != spec.anchor.name for spec in self._concept_specs)):
        return False
    return True