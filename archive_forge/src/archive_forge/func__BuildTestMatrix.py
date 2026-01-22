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
def _BuildTestMatrix(self, spec):
    """Build just the user-specified parts of an iOS TestMatrix message.

    Args:
      spec: a TestSpecification message corresponding to the test type.

    Returns:
      A TestMatrix message.
    """
    devices = [self._BuildIosDevice(d) for d in self._args.device]
    environment_matrix = self._messages.EnvironmentMatrix(iosDeviceList=self._messages.IosDeviceList(iosDevices=devices))
    gcs = self._messages.GoogleCloudStorage(gcsPath=self._gcs_results_root)
    hist = self._messages.ToolResultsHistory(projectId=self._project, historyId=self._history_id)
    results = self._messages.ResultStorage(googleCloudStorage=gcs, toolResultsHistory=hist)
    client_info = matrix_creator_common.BuildClientInfo(self._messages, getattr(self._args, 'client_details', {}) or {}, self._release_track)
    return self._messages.TestMatrix(testSpecification=spec, environmentMatrix=environment_matrix, clientInfo=client_info, resultStorage=results, flakyTestAttempts=self._args.num_flaky_test_attempts or 0)