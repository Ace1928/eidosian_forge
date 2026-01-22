from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def _Synthesize(r):
    """Synthesize a new resource list from the original resource r.

      Args:
        r: The original resource.

      Returns:
        The synthesized resource list.
      """
    synthesized_resource_list = []
    for schema in schemas:
        synthesized_resource = {}
        for attr in schema:
            name, key, literal = attr
            value = resource_property.Get(r, key, None) if key else literal
            if name:
                synthesized_resource[name] = value
            elif isinstance(value, dict):
                synthesized_resource.update(value)
        synthesized_resource_list.append(synthesized_resource)
    return synthesized_resource_list