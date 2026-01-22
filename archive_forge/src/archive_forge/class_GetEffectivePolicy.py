from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class GetEffectivePolicy(base.Command):
    """Get effective policy for a project, folder or organization.

  Get effective policy for a project, folder or organization.

  ## EXAMPLES

   Get effective policy for the current project:

   $ {command}

   Get effective policy for project `my-project`:

   $ {command} --project=my-project
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('--view', help='The view of the effective policy. BASIC includes basic metadata about the effective policy. FULL includes every information related to effective policy.', default='BASIC')
        common_flags.add_resource_args(parser)
        parser.display_info.AddFormat('\n          table(\n            EnabledService:label=EnabledService:sort=1,\n            EnabledPolicies:label=EnabledPolicies\n          )\n        ')

    def Run(self, args):
        """Run command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Effective Policy.
    """
        if args.view not in ('BASIC', 'FULL'):
            raise exceptions.ConfigError('Invalid view. Please provide a valid view. Excepted view : BASIC, FULL')
        if args.IsSpecified('folder'):
            resource_name = _FOLDER_RESOURCE.format(args.folder)
        elif args.IsSpecified('organization'):
            resource_name = _ORGANIZATION_RESOURCE.format(args.organization)
        elif args.IsSpecified('project'):
            resource_name = _PROJECT_RESOURCE.format(args.project)
        else:
            project = properties.VALUES.core.project.Get(required=True)
            resource_name = _PROJECT_RESOURCE.format(project)
        response = serviceusage.GetEffectivePolicyV2Alpha(resource_name + '/effectivePolicy', args.view)
        if args.IsSpecified('format'):
            return response
        else:
            log.status.Print('EnabledRules:')
            for enable_rule in response.enableRules:
                log.status.Print(' Services:')
                for service in enable_rule.services:
                    log.status.Print('  - %s' % service)
            if args.view == 'FULL':
                log.status.Print('\nMetadata of effective policy:')
                result = []
                resources = collections.namedtuple('serviceSources', ['EnabledService', 'EnabledPolicies'])
                for metadata in response.enableRuleMetadata:
                    for values in metadata.serviceSources.additionalProperties:
                        result.append(resources(values.key, values.value.policies))
                return result