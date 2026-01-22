from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServerNetworkTemplate(_messages.Message):
    """Network template.

  Fields:
    applicableInstanceTypes: Instance types this template is applicable to.
    logicalInterfaces: Logical interfaces.
    name: Output only. Template's unique name. The full resource name follows
      the pattern: `projects/{project}/locations/{location}/serverNetworkTempl
      ate/{server_network_template}` Generally, the {server_network_template}
      follows the syntax of "bond" or "nic".
  """
    applicableInstanceTypes = _messages.StringField(1, repeated=True)
    logicalInterfaces = _messages.MessageField('GoogleCloudBaremetalsolutionV2ServerNetworkTemplateLogicalInterface', 2, repeated=True)
    name = _messages.StringField(3)