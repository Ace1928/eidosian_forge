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
def _GetClearedFieldsForUrlRewrite(url_rewrite, field_prefix):
    """Gets a list of fields cleared by the user for UrlRewrite."""
    cleared_fields = []
    if not url_rewrite.pathPrefixRewrite:
        cleared_fields.append(field_prefix + 'pathPrefixRewrite')
    if not url_rewrite.hostRewrite:
        cleared_fields.append(field_prefix + 'hostRewrite')
    return cleared_fields