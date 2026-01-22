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
def _GetClearedFieldsForRoutAction(route_action, field_prefix):
    """Gets a list of fields cleared by the user for HttpRouteAction."""
    cleared_fields = []
    if not route_action.weightedBackendServices:
        cleared_fields.append(field_prefix + 'weightedBackendServices')
    if not route_action.urlRewrite:
        cleared_fields.append(field_prefix + 'urlRewrite')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForUrlRewrite(route_action.urlRewrite, field_prefix + 'urlRewrite.')
    if not route_action.timeout:
        cleared_fields.append(field_prefix + 'timeout')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForDuration(route_action.timeout, field_prefix + 'timeout.')
    if not route_action.retryPolicy:
        cleared_fields.append(field_prefix + 'retryPolicy')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForRetryPolicy(route_action.retryPolicy, field_prefix + 'retryPolicy.')
    if not route_action.requestMirrorPolicy:
        cleared_fields.append(field_prefix + 'requestMirrorPolicy')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForRequestMirrorPolicy(route_action.requestMirrorPolicy, field_prefix + 'requestMirrorPolicy.')
    if not route_action.corsPolicy:
        cleared_fields.append(field_prefix + 'corsPolicy')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForCorsPolicy(route_action.corsPolicy, field_prefix + 'corsPolicy.')
    if not route_action.faultInjectionPolicy:
        cleared_fields.append(field_prefix + 'faultInjectionPolicy')
    else:
        cleared_fields = cleared_fields + _GetClearedFieldsForFaultInjectionPolicy(route_action.faultInjectionPolicy, field_prefix + 'faultInjectionPolicy.')
    return cleared_fields