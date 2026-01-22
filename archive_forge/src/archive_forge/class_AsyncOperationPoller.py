from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
class AsyncOperationPoller(waiter.OperationPoller):
    """An implementation of a operation poller."""

    def __init__(self, spec, resource_ref, args, operation_collection, method):
        """Creates the poller.

    Args:
      spec: yaml_command_schema.CommandData, the spec for the command being
        generated.
      resource_ref: resources.Resource, The resource reference for the resource
        being operated on (not the operation itself). If None, the operation
        will just be returned when it is done instead of getting the resulting
        resource.
      args: Namespace, The args namespace.
      operation_collection: str, collection name of operation
      method: registry.APIMethod, method used to make original api request
    """
        self.spec = spec
        self.args = args
        if not self.spec.async_.extract_resource_result:
            self.resource_ref = None
        else:
            self.resource_ref = resource_ref
        self._operation_collection = operation_collection
        self._resource_collection = method and method.collection.full_name

    @property
    def operation_method(self):
        api_version = self.spec.async_.api_version or self.spec.request.api_version
        return registry.GetMethod(self._operation_collection, self.spec.async_.method, api_version=api_version)

    @property
    def resource_get_method(self):
        return registry.GetMethod(self._resource_collection, 'get', api_version=self.spec.request.api_version)

    def IsDone(self, operation):
        """Overrides."""
        result = getattr(operation, self.spec.async_.state.field)
        if isinstance(result, apitools_messages.Enum):
            result = result.name
        if result in self.spec.async_.state.success_values or result in self.spec.async_.state.error_values:
            error = getattr(operation, self.spec.async_.error.field)
            if not error and result in self.spec.async_.state.error_values:
                error = 'The operation failed.'
            if error:
                raise waiter.OperationError(SerializeError(error))
            return True
        return False

    def Poll(self, operation_ref):
        """Overrides.

    Args:
      operation_ref: googlecloudsdk.core.resources.Resource.

    Returns:
      fetched operation message.
    """
        request_type = self.operation_method.GetRequestType()
        relative_name = operation_ref.RelativeName()
        fields = {}
        for f in request_type.all_fields():
            fields[f.name] = getattr(operation_ref, self.spec.async_.operation_get_method_params.get(f.name, f.name), relative_name)
        request = request_type(**fields)
        for hook in self.spec.async_.modify_request_hooks:
            request = hook(operation_ref, self.args, request)
        return self.operation_method.Call(request)

    def GetResult(self, operation):
        """Overrides.

    Args:
      operation: api_name_messages.Operation.

    Returns:
      result of result_service.Get request.
    """
        result = operation
        if self.resource_ref:
            get_method = self.resource_get_method
            request = get_method.GetRequestType()()
            arg_utils.ParseResourceIntoMessage(self.resource_ref, get_method, request, is_primary_resource=True)
            result = get_method.Call(request)
        return _GetAttribute(result, self.spec.async_.result_attribute)