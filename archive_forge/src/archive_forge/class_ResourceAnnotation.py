from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceAnnotation(_messages.Message):
    """Resource level annotation.

  Fields:
    label: A description of the annotation record.
  """
    label = _messages.StringField(1)