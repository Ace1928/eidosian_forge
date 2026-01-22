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
def _BuildIosXcTestSpec(self):
    """Build a TestSpecification for an IosXcTest."""
    spec = self._messages.TestSpecification(disableVideoRecording=not self._args.record_video, iosTestSetup=self._BuildGenericTestSetup(), testTimeout=matrix_ops.ReformatDuration(self._args.timeout), iosXcTest=self._messages.IosXcTest(testsZip=self._BuildFileReference(self._args.test), xctestrun=self._BuildFileReference(self._args.xctestrun_file), xcodeVersion=self._args.xcode_version, testSpecialEntitlements=getattr(self._args, 'test_special_entitlements', False)))
    return spec