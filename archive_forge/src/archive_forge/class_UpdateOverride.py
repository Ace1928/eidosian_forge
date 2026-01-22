from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profiles.threat_prevention import sp_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_security import sp_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class UpdateOverride(base.UpdateCommand):
    """Update Overrides of Threat Prevention Profile."""

    @classmethod
    def Args(cls, parser):
        sp_flags.AddSecurityProfileResource(parser, cls.ReleaseTrack())
        sp_flags.AddSeverityorThreatIDArg(parser, required=True)
        sp_flags.AddActionArg(parser, required=True)
        labels_util.AddUpdateLabelsFlags(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        base.ASYNC_FLAG.SetDefault(parser, False)

    def getLabel(self, client, security_profile):
        return client.GetSecurityProfile(security_profile.RelativeName()).labels

    def Run(self, args):
        client = sp_api.Client(self.ReleaseTrack())
        security_profile = args.CONCEPTS.security_profile.Parse()
        is_async = args.async_
        labels_update = labels_util.ProcessUpdateArgsLazy(args, client.messages.SecurityProfile.LabelsValue, orig_labels_thunk=lambda: self.getLabel(client, security_profile))
        overrides = []
        if not args.IsSpecified('action'):
            raise core_exceptions.Error('--action must be specified')
        if args.IsSpecified('severities'):
            update_mask = 'severityOverrides'
            severities = args.severities
            action = args.action
            for severity in severities:
                overrides.append({'severity': severity, 'action': action})
        elif args.IsSpecified('threat_ids'):
            update_mask = 'threatOverrides'
            threats = args.threat_ids
            action = args.action
            for threat in threats:
                overrides.append({'threatId': threat, 'action': action})
        else:
            raise core_exceptions.Error('Either --severities or --threat-ids must be specified')
        if args.location != 'global':
            raise core_exceptions.Error('Only `global` location is supported, but got: %s' % args.location)
        response = client.ModifyOverride(security_profile.RelativeName(), overrides, 'update_override', update_mask, labels=labels_update.GetOrNone())
        if is_async:
            operation_id = response.name
            log.status.Print('Check for operation completion status using operation ID:', operation_id)
            return response
        return client.WaitForOperation(operation_ref=client.GetOperationsRef(response), message='Waiting for update override in security-profile [{}] operation to complete.'.format(security_profile.RelativeName()), has_result=True)