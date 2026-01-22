from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _BuildGenericTestSetup(self):
    """Build an IosTestSetup for an iOS test."""
    additional_ipas = [self._BuildFileReference(os.path.basename(additional_ipa)) for additional_ipa in getattr(self._args, 'additional_ipas', []) or []]
    directories_to_pull = []
    for directory in getattr(self._args, 'directories_to_pull', []) or []:
        if ':' in directory:
            bundle, path = directory.split(':')
            directories_to_pull.append(self._messages.IosDeviceFile(bundleId=bundle, devicePath=path))
        else:
            directories_to_pull.append(self._messages.IosDeviceFile(devicePath=directory))
    device_files = []
    other_files = getattr(self._args, 'other_files', None) or {}
    for device_path in other_files.keys():
        idx = device_path.find(':')
        bundle_id = device_path[:idx] if idx != -1 else None
        path = device_path[idx + 1:] if idx != -1 else device_path
        device_files.append(self._messages.IosDeviceFile(content=self._BuildFileReference(util.GetRelativeDevicePath(path), use_basename=False), bundleId=bundle_id, devicePath=path))
    return self._messages.IosTestSetup(networkProfile=getattr(self._args, 'network_profile', None), additionalIpas=additional_ipas, pushFiles=device_files, pullDirectories=directories_to_pull)