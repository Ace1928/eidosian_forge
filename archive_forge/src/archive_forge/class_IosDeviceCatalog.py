from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosDeviceCatalog(_messages.Message):
    """The currently supported iOS devices.

  Fields:
    models: The set of supported iOS device models.
    runtimeConfiguration: The set of supported runtime configurations.
    versions: The set of supported iOS software versions.
    xcodeVersions: The set of supported Xcode versions.
  """
    models = _messages.MessageField('IosModel', 1, repeated=True)
    runtimeConfiguration = _messages.MessageField('IosRuntimeConfiguration', 2)
    versions = _messages.MessageField('IosVersion', 3, repeated=True)
    xcodeVersions = _messages.MessageField('XcodeVersion', 4, repeated=True)