from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
class EnvironmentPoller(waiter.OperationPoller):
    """Poller for environment operations."""

    def __init__(self, environments_service, operations_service):
        self.environments_service = environments_service
        self.operations_service = operations_service

    def IsDone(self, operation):
        return operation.done

    def Poll(self, operation):
        request_type = self.operations_service.GetRequestType('Get')
        return self.operations_service.Get(request_type(name=operation.name))

    def GetResult(self, operation):
        request_type = self.environments_service.GetRequestType('Get')
        return self.environments_service.Get(request_type(name=DEFAULT_ENVIRONMENT_NAME))