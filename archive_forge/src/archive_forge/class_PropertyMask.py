from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PropertyMask(_messages.Message):
    """The set of arbitrarily nested property paths used to restrict an
  operation to only a subset of properties in an entity.

  Fields:
    paths: The paths to the properties covered by this mask. A path is a list
      of property names separated by dots (`.`), for example `foo.bar` means
      the property `bar` inside the entity property `foo` inside the entity
      associated with this path. If a property name contains a dot `.` or a
      backslash `\\`, then that name must be escaped. A path must not be empty,
      and may not reference a value inside an array value.
  """
    paths = _messages.StringField(1, repeated=True)