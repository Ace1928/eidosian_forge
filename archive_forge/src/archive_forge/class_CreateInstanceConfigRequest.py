from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateInstanceConfigRequest(_messages.Message):
    """The request for CreateInstanceConfigRequest.

  Fields:
    instanceConfig: Required. The InstanceConfig proto of the configuration to
      create. instance_config.name must be `/instanceConfigs/`.
      instance_config.base_config must be a Google managed configuration name,
      e.g. /instanceConfigs/us-east1, /instanceConfigs/nam3.
    instanceConfigId: Required. The ID of the instance config to create. Valid
      identifiers are of the form `custom-[-a-z0-9]*[a-z0-9]` and must be
      between 2 and 64 characters in length. The `custom-` prefix is required
      to avoid name conflicts with Google managed configurations.
    validateOnly: An option to validate, but not actually execute, a request,
      and provide the same response.
  """
    instanceConfig = _messages.MessageField('InstanceConfig', 1)
    instanceConfigId = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)