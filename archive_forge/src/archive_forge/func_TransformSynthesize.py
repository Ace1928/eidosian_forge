from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformSynthesize(r, *args):
    """Synthesizes a new resource from the schema arguments.

  A list of tuple arguments controls the resource synthesis. Each tuple is a
  schema that defines the synthesis of one resource list item. Each schema
  item defines the synthesis of one synthesized_resource attribute from an
  original_resource attribute.

  There are three kinds of schema items:

  *name:literal*:::
  The value for the name attribute in the synthesized resource is the literal
  value.
  *name=key*:::
  The value for the name attribute in the synthesized_resource is the
  value of key in the original_resource.
  *key*:::
  All the attributes of the value of key in the original_resource are
  added to the attributes in the synthesized_resource.
  :::

  Args:
    r: A resource list.
    *args: The list of schema tuples.

  Example:
    This returns a list of two resource items:::
    `synthesize((name:up, upInfo), (name:down, downInfo))`
    If upInfo and downInfo serialize to:::
    `{"foo": 1, "bar": "yes"}`
    and:::
    `{"foo": 0, "bar": "no"}`
    then the synthesized resource list is:::
    `[{"name": "up", "foo": 1, "bar": "yes"},
      {"name": "down", "foo": 0, "bar": "no"}]`
    This could then be displayed by a nested table using:::
    `synthesize(...):format="table(name, foo, bar)"`


  Returns:
    A synthesized resource list.
  """
    _ = args
    return r