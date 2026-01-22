from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def UpdateAuditLogsEnabled(unused_domain_ref, args, patch_request):
    """Updates audit logs config for the domain."""
    if args.IsSpecified('enable_audit_logs'):
        patch_request.domain.auditLogsEnabled = args.enable_audit_logs
        AddFieldToUpdateMask('audit_logs_enabled', patch_request)
    return patch_request