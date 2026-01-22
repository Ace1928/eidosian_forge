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
def _BuildIosRoboTestSpec(self):
    """Build a TestSpecification for an iOS Robo test."""
    spec = self._messages.TestSpecification(disableVideoRecording=not self._args.record_video, iosTestSetup=self._BuildGenericTestSetup(), testTimeout=matrix_ops.ReformatDuration(self._args.timeout), iosRoboTest=self._messages.IosRoboTest(appIpa=self._BuildFileReference(self._args.app)))
    if getattr(self._args, 'robo_script', None):
        spec.iosRoboTest.roboScript = self._BuildFileReference(os.path.basename(self._args.robo_script))
    return spec