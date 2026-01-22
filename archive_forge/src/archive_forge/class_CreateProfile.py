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
class CreateProfile(base.CreateCommand):
    """Create a new Threat Prevention Profile."""

    @classmethod
    def Args(cls, parser):
        sp_flags.AddSecurityProfileResource(parser, cls.ReleaseTrack())
        sp_flags.AddProfileDescription(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        base.ASYNC_FLAG.SetDefault(parser, False)
        labels_util.AddCreateLabelsFlags(parser)

    def Run(self, args):
        client = sp_api.Client(self.ReleaseTrack())
        security_profile = args.CONCEPTS.security_profile.Parse()
        description = args.description
        labels = labels_util.ParseCreateArgs(args, client.messages.SecurityProfile.LabelsValue)
        is_async = args.async_
        if not args.IsSpecified('description'):
            args.description = 'Security Profile of type Threat Prevention'
        if args.location != 'global':
            raise core_exceptions.Error('Only `global` location is supported, but got: %s' % args.location)
        response = client.CreateSecurityProfile(name=security_profile.RelativeName(), sp_id=security_profile.Name(), parent=security_profile.Parent().RelativeName(), description=description, labels=labels)
        if is_async:
            operation_id = response.name
            log.status.Print('Check for operation completion status using operation ID:', operation_id)
            return response
        return client.WaitForOperation(operation_ref=client.GetOperationsRef(response), message='Waiting for security-profile [{}] to be created'.format(security_profile.RelativeName()), has_result=True)