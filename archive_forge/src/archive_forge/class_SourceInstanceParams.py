from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceInstanceParams(_messages.Message):
    """A specification of the parameters to use when creating the instance
  template from a source instance.

  Fields:
    diskConfigs: Attached disks configuration. If not provided, defaults are
      applied: For boot disk and any other R/W disks, the source images for
      each disk will be used. For read-only disks, they will be attached in
      read-only mode. Local SSD disks will be created as blank volumes.
  """
    diskConfigs = _messages.MessageField('DiskInstantiationConfig', 1, repeated=True)