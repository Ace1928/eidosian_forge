from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1ApiTarget(_messages.Message):
    """A restriction for a specific service and optionally one or multiple
  specific methods. Both fields are not case sensitive.

  Fields:
    methods: Optional. List of one or more methods that can be called. If
      empty, all methods for the service are allowed. A wildcard (*) can be
      used as the last symbol. Valid examples:
      google.cloud.translate.v2.TranslateService.GetSupportedLanguage
      TranslateText Get* google.cloud.translate.v2.TranslateService.Get*
    service: The service for this restriction. It should be canonical One
      Platform service name, for example:
      google.cloud.translate.v2.TranslateService.
  """
    methods = _messages.StringField(1, repeated=True)
    service = _messages.StringField(2)