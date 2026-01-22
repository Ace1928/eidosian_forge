from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespaceSpec(_messages.Message):
    """Not supported by Cloud Run. NamespaceSpec describes the attributes on a
  Namespace.

  Fields:
    finalizers: Finalizers is an opaque list of values that must be empty to
      permanently remove object from storage.
  """
    finalizers = _messages.StringField(1, repeated=True)