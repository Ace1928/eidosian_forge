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
def _HandleAsync(self, args, resource_ref, operation, request_string, extract_resource_result=True):
    """Handles polling for operations if the async flag is provided.

    Args:
      args: argparse.Namespace, The parsed args.
      resource_ref: resources.Resource, The resource reference for the resource
        being operated on (not the operation itself)
      operation: The operation message response.
      request_string: The format string to print indicating a request has been
        issued for the resource. If None, nothing is printed.
      extract_resource_result: bool, True to return the original resource as
        the result or False to just return the operation response when it is
        done. You would set this to False for things like Delete where the
        resource no longer exists when the operation is done.

    Returns:
      The response (either the operation or the original resource).
    """
    operation_ref, operation_collection = self._GetOperationRef(operation)
    request_string = self.spec.async_.request_issued_message or request_string
    if request_string:
        log.status.Print(self._Format(request_string, resource_ref, self._GetDisplayResourceType(args), self._GetDisplayName(resource_ref, args)))
    if args.async_:
        log.status.Print(self._Format('Check operation [{{{}}}] for status.'.format(yaml_command_schema_util.REL_NAME_FORMAT_KEY), operation_ref, self._GetDisplayResourceType(args)))
        return operation
    method = self.arg_generator.GetPrimaryResource(self.methods, args).method
    poller = AsyncOperationPoller(self.spec, resource_ref if extract_resource_result else None, args, operation_collection, method)
    if poller.IsDone(operation):
        return poller.GetResult(operation)
    return self._WaitForOperationWithPoller(poller, operation_ref, args=args)