from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespaceStatus(_messages.Message):
    """Not supported by Cloud Run. NamespaceStatus is information about the
  current status of a Namespace.

  Fields:
    phase: Phase is the current lifecycle phase of the namespace.
  """
    phase = _messages.StringField(1)