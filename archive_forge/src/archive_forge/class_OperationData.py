from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.util import exceptions as http_exceptions
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class OperationData(object):
    """Holds all information necessary to poll given operation.

  Attributes:
    operation: An Operation object to poll.
    operation_service: The service that can be used to get operation object.
    resource_service: The service of the collection being mutated by the
      operation. If the operation type is not delete, this service is used to
      fetch the mutated object after the operation is done.
    project: str, The project to which the resource belong.
    resize_request_name: str, Name of the resize request being created.
    no_followup: str, If True, do not send followup GET request.
    followup_override: str, Overrides the target resource name when it is
      different from the resource name which is used to poll.
    always_return_operation: If true, always return operation object even if the
      operation fails.
    errors: An output parameter for capturing errors.
    warnings: An output parameter for capturing warnings.
  """

    def __init__(self, operation, operation_service, resource_service, project=None, resize_request_name=None, no_followup=False, followup_override=None, always_return_operation=False):
        self.operation = operation
        self.operation_service = operation_service
        self.resource_service = resource_service
        self.project = project
        self.resize_request_name = resize_request_name
        self.no_followup = no_followup
        self.followup_override = followup_override
        self.always_return_operation = always_return_operation
        self.errors = []
        self.warnings = []

    def __eq__(self, o):
        if not isinstance(o, OperationData):
            return False
        return self.operation == o.operation and self.project == o.project and (self.operation_service == o.operation_service) and (self.resource_service == o.resource_service) and (self.no_followup == o.no_followup) and (self.followup_override == o.followup_override)

    def __hash__(self):
        return hash(self.operation.selfLink) ^ hash(self.project) ^ hash(self.operation_service) ^ hash(self.resource_service) ^ hash(self.no_followup) ^ hash(self.followup_override)

    def __ne__(self, o):
        return not self == o

    def SetOperation(self, operation):
        """"Updates the operation.

    Args:
      operation: Operation to be assigned.
    """
        self.operation = operation

    def IsGlobalOrganizationOperation(self):
        if not hasattr(self.operation_service.client, 'globalOrganizationOperations'):
            return False
        return self.operation_service == self.operation_service.client.globalOrganizationOperations

    def IsDone(self):
        """Returns true if the operation is done."""
        operation_type = self.operation_service.GetResponseType('Get')
        done = operation_type.StatusValueValuesEnum.DONE
        return self.operation.status == done

    def _SupportOperationWait(self):
        return 'Wait' in self.operation_service.GetMethodsList()

    def ResourceGetRequest(self):
        """"Generates apitools request message to get the resource."""
        target_link = self.operation.targetLink
        if self.project:
            request = self.resource_service.GetRequestType('Get')(project=self.project)
        else:
            if target_link is None:
                log.status.write('{0}.\n'.format(_HumanFriendlyNameForOpPastTense(self.operation.operationType).capitalize()))
                return
            token_list = target_link.split('/')
            flexible_resource_id = token_list[-1]
            request = self.resource_service.GetRequestType('Get')(securityPolicy=flexible_resource_id)
        if self.operation.zone:
            request.zone = path_simplifier.Name(self.operation.zone)
        elif self.operation.region:
            request.region = path_simplifier.Name(self.operation.region)
        resource_params = self.resource_service.GetMethodConfig('Get').ordered_params
        name_field = resource_params[-1]
        if len(resource_params) == 4:
            if self.resize_request_name:
                target_link = target_link + '/resizeRequests/' + self.resize_request_name
            parent_resource_field = resource_params[2]
            parent_resource_name = target_link.split('/')[-3]
            setattr(request, parent_resource_field, parent_resource_name)
        resource_name = self.followup_override or path_simplifier.Name(target_link)
        setattr(request, name_field, resource_name)
        return request

    def _OperationRequest(self, verb):
        """Generates apitools request message to poll the operation."""
        if self.project:
            request = self.operation_service.GetRequestType(verb)(operation=self.operation.name, project=self.project)
        else:
            token_list = self.operation.name.split('-')
            parent_id = 'organizations/' + token_list[1]
            request = self.operation_service.GetRequestType(verb)(operation=self.operation.name, parentId=parent_id)
        if self.operation.zone:
            request.zone = path_simplifier.Name(self.operation.zone)
        elif self.operation.region:
            request.region = path_simplifier.Name(self.operation.region)
        return request

    def OperationGetRequest(self):
        """Generates apitools request message for operations.get method."""
        return self._OperationRequest('Get')

    def OperationWaitRequest(self):
        """Generates apitools request message for operations.wait method."""
        return self._OperationRequest('Wait')

    def _CallService(self, method, request):
        try:
            return method(request)
        except apitools_exceptions.HttpError as e:
            http_err = http_exceptions.HttpException(e)
            self.errors.append((http_err.error.status_code, http_err.message))
            _RecordProblems(self.operation, self.warnings, self.errors)
            raise

    def _PollUntilDoneUsingOperationGet(self, timeout_sec=_POLLING_TIMEOUT_SEC):
        """Polls the operation with operation Get method."""
        get_request = self.OperationGetRequest()
        start = time_util.CurrentTimeSec()
        poll_time_interval = 0
        max_poll_interval = 5
        while True:
            if time_util.CurrentTimeSec() - start > timeout_sec:
                self.errors.append((None, 'operation {} timed out'.format(self.operation.name)))
                _RecordProblems(self.operation, self.warnings, self.errors)
                return
            try:
                self.operation = self._CallService(self.operation_service.Get, get_request)
            except apitools_exceptions.HttpError:
                return
            if self.IsDone():
                _RecordProblems(self.operation, self.warnings, self.errors)
                return
            poll_time_interval = min(poll_time_interval + 1, max_poll_interval)
            time_util.Sleep(poll_time_interval)

    def _PollUntilDoneUsingOperationWait(self, timeout_sec=_POLLING_TIMEOUT_SEC):
        """Polls the operation with operation method."""
        wait_request = self.OperationWaitRequest()
        start = time_util.CurrentTimeSec()
        while not self.IsDone():
            if time_util.CurrentTimeSec() - start > timeout_sec:
                self.errors.append((None, 'operation {} timed out'.format(self.operation.name)))
                _RecordProblems(self.operation, self.warnings, self.errors)
                return
            try:
                self.operation = self._CallService(self.operation_service.Wait, wait_request)
            except apitools_exceptions.HttpError:
                return
        _RecordProblems(self.operation, self.warnings, self.errors)

    def PollUntilDone(self, timeout_sec=_POLLING_TIMEOUT_SEC):
        """Polls the operation until it is done."""
        if self.IsDone():
            return
        if self._SupportOperationWait():
            self._PollUntilDoneUsingOperationWait(timeout_sec)
        else:
            self._PollUntilDoneUsingOperationGet(timeout_sec)

    def GetResult(self, timeout_sec=_POLLING_TIMEOUT_SEC):
        """Get the resource which is touched by the operation."""
        self.PollUntilDone(timeout_sec)
        if not self.no_followup and (not self.operation.error) and (not _IsDeleteOp(self.operation.operationType)):
            resource_get_request = self.ResourceGetRequest()
            try:
                return self._CallService(self.resource_service.Get, resource_get_request)
            except apitools_exceptions.HttpError:
                pass