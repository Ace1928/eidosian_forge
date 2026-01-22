from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypeInfoSchemaInfo(_messages.Message):
    """A TypeInfoSchemaInfo object.

  Fields:
    input: The properties that this composite type or base type collection
      accept as input, represented as a json blob, format is: JSON Schema
      Draft V4
    output: The properties that this composite type or base type collection
      exposes as output, these properties can be used for references,
      represented as json blob, format is: JSON Schema Draft V4
  """
    input = _messages.StringField(1)
    output = _messages.StringField(2)