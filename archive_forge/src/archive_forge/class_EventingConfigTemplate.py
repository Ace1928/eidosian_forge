from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventingConfigTemplate(_messages.Message):
    """Eventing Config details of a connector version.

  Fields:
    additionalVariables: Additional fields that need to be rendered.
    authConfigTemplates: AuthConfigTemplates represents the auth values for
      the webhook adapter.
    autoRefresh: Auto refresh to extend webhook life.
    autoRegistrationSupported: Auto Registration supported.
    encryptionKeyTemplate: Encryption key (can be either Google managed or
      CMEK).
    enrichmentSupported: Enrichment Supported.
    isEventingSupported: Is Eventing Supported.
    registrationDestinationConfig: Registration host destination config
      template.
  """
    additionalVariables = _messages.MessageField('ConfigVariableTemplate', 1, repeated=True)
    authConfigTemplates = _messages.MessageField('AuthConfigTemplate', 2, repeated=True)
    autoRefresh = _messages.BooleanField(3)
    autoRegistrationSupported = _messages.BooleanField(4)
    encryptionKeyTemplate = _messages.MessageField('ConfigVariableTemplate', 5)
    enrichmentSupported = _messages.BooleanField(6)
    isEventingSupported = _messages.BooleanField(7)
    registrationDestinationConfig = _messages.MessageField('DestinationConfigTemplate', 8)