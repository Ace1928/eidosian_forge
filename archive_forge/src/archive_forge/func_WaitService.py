from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def WaitService(self, operation_id):
    """Return the relevant Service from the server, or None if 404."""
    messages = self.messages_module
    project = properties.VALUES.core.project.Get(required=True)
    op_name = f'projects/{project}/locations/{self._region}/operations/{operation_id}'
    op_ref = self._registry.ParseRelativeName(op_name, collection='run.projects.locations.operations')
    try:
        with metrics.RecordDuration(metric_names.WAIT_OPERATION):
            poller = op_pollers.WaitOperationPoller(self._client.projects_locations_services, self._client.projects_locations_operations)
            operation = waiter.PollUntilDone(poller, op_ref)
            as_dict = encoding.MessageToPyValue(operation.response)
            as_pb = encoding.PyValueToMessage(messages.Service, as_dict)
            return service.Service(as_pb, self.messages_module)
    except api_exceptions.InvalidDataFromServerError as e:
        serverless_exceptions.MaybeRaiseCustomFieldMismatch(e)
    except api_exceptions.HttpNotFoundError:
        return None