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
def _BuildAndroidGameLoopTestSpec(self):
    """Build a TestSpecification for an AndroidTestLoop."""
    spec = self._BuildGenericTestSpec()
    app_apk, app_bundle = self._BuildAppReference(self._args.app)
    spec.androidTestLoop = self._messages.AndroidTestLoop(appApk=app_apk, appBundle=app_bundle, appPackageId=self._args.app_package)
    if self._args.scenario_numbers:
        spec.androidTestLoop.scenarios = self._args.scenario_numbers
    if self._args.scenario_labels:
        spec.androidTestLoop.scenarioLabels = self._args.scenario_labels
    return spec