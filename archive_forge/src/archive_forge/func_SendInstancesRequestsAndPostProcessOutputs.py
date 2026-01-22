from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def SendInstancesRequestsAndPostProcessOutputs(api_holder, method_name, request_template, instances_holder_field, igm_ref, instances, per_instance_status_enabled=False):
    """Make *-instances requests and format output.

  Method resolves instance references, splits them to make batch of requests,
  adds to results statuses for unresolved instances, and yields all statuses
  raising errors, if any, in the end.

  Args:
    api_holder: Compute API holder.
    method_name: Name of the (region) instance groups managers service method to
      call.
    request_template: Partially filled *-instances request (no instances).
    instances_holder_field: Name of the field inside request holding instances
      field.
    igm_ref: URL to the target IGM.
    instances: A list of names of the instances to apply method to.
    per_instance_status_enabled: Enable functionality parsing resulting
      operation for graceful validation related warnings to allow per-instance
      status output. The plan is to gradually enable this for all per-instance
      commands in GA (even where graceful validation is not available / not
      used).

  Yields:
    A list of request statuses per instance. Requests status is a dictionary
    object with link to an instance keyed with 'selfLink', instance name keyed
    with 'instanceName', and status indicating if operation succeeded for
    instance keyed with 'status'. Status might be 'FAIL', 'SUCCESS', 'SKIPPED'
    in case of graceful validation, or 'MEMBER_NOT_FOUND' (in case of regional
    MIGs, when instance name cannot be resolved).
  """
    client = api_holder.client
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        service = client.apitools_client.instanceGroupManagers
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        service = client.apitools_client.regionInstanceGroupManagers
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
    instances_with_references = CreateInstanceReferences(api_holder.resources, client, igm_ref, instances)
    resolved_references = [instance.instance_reference for instance in instances_with_references if instance.instance_reference]
    getattr(request_template, instances_holder_field).instances = resolved_references
    requests = SplitInstancesInRequest(request_template, instances_holder_field)
    request_tuples = GenerateRequestTuples(service, method_name, requests)
    errors_to_collect = []
    warnings_to_collect = []
    request_status_per_instance = []
    if per_instance_status_enabled:
        request_status_per_instance.extend(MakeRequestsAndGetStatusPerInstanceFromOperation(client, request_tuples, instances_holder_field, warnings_to_collect, errors_to_collect))
    else:
        request_status_per_instance.extend(MakeRequestsAndGetStatusPerInstance(client, request_tuples, instances_holder_field, errors_to_collect))
    unresolved_instance_names = [instance.instance_name for instance in instances_with_references if not instance.instance_reference]
    request_status_per_instance.extend([dict(instanceName=name, status='MEMBER_NOT_FOUND') for name in unresolved_instance_names])
    for status in request_status_per_instance:
        yield status
    if warnings_to_collect:
        log.warning(utils.ConstructList('Some requests generated warnings:', warnings_to_collect))
    if errors_to_collect:
        raise utils.RaiseToolException(errors_to_collect)