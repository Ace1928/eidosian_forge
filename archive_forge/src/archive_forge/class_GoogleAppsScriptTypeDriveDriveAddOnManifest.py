from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeDriveDriveAddOnManifest(_messages.Message):
    """Properties customizing the appearance and execution of a Drive add-on.

  Fields:
    homepageTrigger: If present, this overrides the configuration from
      `addOns.common.homepageTrigger`.
    onItemsSelectedTrigger: Corresponds to behavior that executes when items
      are selected in the relevant Drive view, such as the My Drive Doclist.
  """
    homepageTrigger = _messages.MessageField('GoogleAppsScriptTypeHomepageExtensionPoint', 1)
    onItemsSelectedTrigger = _messages.MessageField('GoogleAppsScriptTypeDriveDriveExtensionPoint', 2)