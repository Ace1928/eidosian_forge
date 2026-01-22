from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeSlidesSlidesAddOnManifest(_messages.Message):
    """Properties customizing the appearance and execution of a Google Slides
  add-on.

  Fields:
    homepageTrigger: If present, this overrides the configuration from
      `addOns.common.homepageTrigger`.
    linkPreviewTriggers: A list of extension points for previewing links in a
      Google Slides document. For details, see [Preview links with smart
      chips](https://developers.google.com/workspace/add-ons/guides/preview-
      links-smart-chips).
    onFileScopeGrantedTrigger: Endpoint to execute when file scope
      authorization is granted for this document/user pair.
  """
    homepageTrigger = _messages.MessageField('GoogleAppsScriptTypeHomepageExtensionPoint', 1)
    linkPreviewTriggers = _messages.MessageField('GoogleAppsScriptTypeLinkPreviewExtensionPoint', 2, repeated=True)
    onFileScopeGrantedTrigger = _messages.MessageField('GoogleAppsScriptTypeSlidesSlidesExtensionPoint', 3)