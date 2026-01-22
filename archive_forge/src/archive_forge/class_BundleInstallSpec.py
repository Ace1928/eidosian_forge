from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BundleInstallSpec(_messages.Message):
    """BundleInstallSpec is the specification configuration for a single
  managed bundle.

  Fields:
    exemptedNamespaces: The set of namespaces to be exempted from the bundle.
  """
    exemptedNamespaces = _messages.StringField(1, repeated=True)