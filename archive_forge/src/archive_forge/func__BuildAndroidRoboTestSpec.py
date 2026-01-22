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
def _BuildAndroidRoboTestSpec(self):
    """Build a TestSpecification for an AndroidRoboTest."""
    spec = self._BuildGenericTestSpec()
    app_apk, app_bundle = self._BuildAppReference(self._args.app)
    robo_modes = self._messages.AndroidRoboTest.RoboModeValueValuesEnum
    robo_mode = robo_modes.ROBO_VERSION_2 if getattr(self._args, 'resign', True) else robo_modes.ROBO_VERSION_1
    spec.androidRoboTest = self._messages.AndroidRoboTest(appApk=app_apk, appBundle=app_bundle, appPackageId=self._args.app_package, roboDirectives=self._BuildRoboDirectives(self._args.robo_directives), roboMode=robo_mode)
    if getattr(self._args, 'robo_script', None):
        spec.androidRoboTest.roboScript = self._BuildFileReference(os.path.basename(self._args.robo_script))
    return spec