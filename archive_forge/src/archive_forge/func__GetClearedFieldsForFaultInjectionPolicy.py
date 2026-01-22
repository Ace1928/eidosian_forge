from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def _GetClearedFieldsForFaultInjectionPolicy(fault_injection_policy, field_prefix):
    """Gets a list of fields cleared by the user for FaultInjectionPolicy."""
    cleared_fields = []
    if not fault_injection_policy.delay:
        cleared_fields.append(field_prefix + 'delay')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForFaultDelay(fault_injection_policy.delay, field_prefix + 'delay.')
    if not fault_injection_policy.abort:
        cleared_fields.append(field_prefix + 'abort')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForFaultAbort(fault_injection_policy.abort, field_prefix + 'abort.')
    return cleared_fields