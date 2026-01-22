from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import operator
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import exceptions
from googlecloudsdk.api_lib.app import instances_util
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import region_util
from googlecloudsdk.api_lib.app import service_util
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.third_party.appengine.admin.tools.conversion import convert_yaml
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
def RepairApplication(self, progress_message=None):
    """Creates missing app resources.

    In particular, the Application.code_bucket GCS reference.

    Args:
      progress_message: str, the message to use while the operation is polled,
        if not the default.

    Returns:
      A long running operation.
    """
    request = self.messages.AppengineAppsRepairRequest(name=self._FormatApp(), repairApplicationRequest=self.messages.RepairApplicationRequest())
    operation = self.client.apps.Repair(request)
    log.debug('Received operation: [{operation}]'.format(operation=operation.name))
    return operations_util.WaitForOperation(self.client.apps_operations, operation, message=progress_message)