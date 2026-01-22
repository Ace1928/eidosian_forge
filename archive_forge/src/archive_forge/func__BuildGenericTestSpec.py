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
import six
def _BuildGenericTestSpec(self):
    """Build a generic TestSpecification without test-type specifics."""
    device_files = []
    for obb_file in self._args.obb_files or []:
        obb_file_name = os.path.basename(obb_file)
        device_files.append(self._messages.DeviceFile(obbFile=self._messages.ObbFile(obbFileName=obb_file_name, obb=self._BuildFileReference(obb_file_name))))
    other_files = getattr(self._args, 'other_files', None) or {}
    for device_path in other_files.keys():
        device_files.append(self._messages.DeviceFile(regularFile=self._messages.RegularFile(content=self._BuildFileReference(util.GetRelativeDevicePath(device_path)), devicePath=device_path)))
    environment_variables = []
    if self._args.environment_variables:
        for key, value in six.iteritems(self._args.environment_variables):
            environment_variables.append(self._messages.EnvironmentVariable(key=key, value=value))
    directories_to_pull = self._args.directories_to_pull or []
    account = None
    if self._args.auto_google_login:
        account = self._messages.Account(googleAuto=self._messages.GoogleAuto())
    additional_apks = [self._messages.Apk(location=self._BuildFileReference(os.path.basename(additional_apk))) for additional_apk in getattr(self._args, 'additional_apks', []) or []]
    grant_permissions = getattr(self._args, 'grant_permissions', 'all') == 'all'
    setup = self._messages.TestSetup(filesToPush=device_files, account=account, environmentVariables=environment_variables, directoriesToPull=directories_to_pull, networkProfile=getattr(self._args, 'network_profile', None), additionalApks=additional_apks, dontAutograntPermissions=not grant_permissions)
    return self._messages.TestSpecification(testTimeout=matrix_ops.ReformatDuration(self._args.timeout), testSetup=setup, disableVideoRecording=not self._args.record_video, disablePerformanceMetrics=not self._args.performance_metrics)