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
def UpdateTraffic(self, service_ref, config_changes, tracker, asyn):
    """Update traffic splits for service."""
    if tracker is None:
        tracker = progress_tracker.NoOpStagedProgressTracker(stages.UpdateTrafficStages(), interruptable=True, aborted_message='aborted')
    serv = self.GetService(service_ref)
    if not serv:
        raise serverless_exceptions.ServiceNotFoundError('Service [{}] could not be found.'.format(service_ref.servicesId))
    updated_serv = self._UpdateOrCreateService(service_ref, config_changes, False, serv)
    if not asyn:
        if updated_serv.conditions.IsReady():
            return updated_serv
        getter = functools.partial(self.GetService, service_ref) if updated_serv.operation_id is None else functools.partial(self.WaitService, updated_serv.operation_id)
        poller = op_pollers.ServiceConditionPoller(getter, tracker, serv=updated_serv)
        self.WaitForCondition(poller)
        updated_serv = poller.GetResource()
    return updated_serv