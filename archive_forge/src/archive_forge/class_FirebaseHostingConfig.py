from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebaseHostingConfig(_messages.Message):
    """Message for defining firebase hosting resource.

  Fields:
    config: Hosting site configuration.
    resources: Reference to the target resources to add to the hosting site
      configuration. The resource binding's "binding-config" has no used
      fields currently.
  """
    config = _messages.MessageField('HostingSiteConfig', 1)
    resources = _messages.MessageField('ServiceResourceBindingConfig', 2, repeated=True)