from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
def ModifyConfig(self, device_ref, data, version=None):
    """Modify a device configuration.

    Follows the API semantics, notably those regarding the version parameter: If
    0 or None, the latest version is modified. Otherwise, this update will fail
    if the version number provided does not match the latest version on the
    server.

    Args:
      device_ref: a Resource reference to a
        cloudiot.projects.locations.registries.devices resource.
      data: str, the binary data for the configuration
      version: int or None, the version of the configuration to modify.

    Returns:
      DeviceConfig: the modified DeviceConfig for the device
    """
    request_type = getattr(self.messages, 'CloudiotProjectsLocationsRegistriesDevicesModifyCloudToDeviceConfigRequest')
    modify_request_type = self.messages.ModifyCloudToDeviceConfigRequest
    request = request_type(name=device_ref.RelativeName(), modifyCloudToDeviceConfigRequest=modify_request_type(binaryData=data, versionToUpdate=version))
    return self._service.ModifyCloudToDeviceConfig(request)