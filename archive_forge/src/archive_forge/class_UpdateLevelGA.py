from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import levels as levels_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateLevelGA(base.UpdateCommand):
    """Update an existing access level."""
    _API_VERSION = _API_VERSION_PER_TRACK.get('GA')
    _FEATURE_MASK = _FEATURE_MASK_PER_TRACK.get('GA')

    @staticmethod
    def Args(parser):
        UpdateLevelGA.ArgsVersioned(parser, release_track='GA')

    @staticmethod
    def ArgsVersioned(parser, release_track):
        api_version = _API_VERSION_PER_TRACK.get(release_track, 'v1')
        feature_mask = _FEATURE_MASK_PER_TRACK.get(release_track, {})
        levels.AddResourceArg(parser, 'to update')
        levels.AddLevelArgs(parser)
        levels.AddLevelSpecArgs(parser, api_version=api_version, feature_mask=feature_mask)

    def Run(self, args):
        client = levels_api.Client(version=self._API_VERSION)
        level_ref = args.CONCEPTS.level.Parse()
        policies.ValidateAccessPolicyArg(level_ref, args)
        basic_level_combine_function = None
        if args.IsSpecified('combine_function'):
            mapper = levels.GetCombineFunctionEnumMapper(api_version=self._API_VERSION)
            basic_level_combine_function = mapper.GetEnumForChoice(args.combine_function)
        custom_level_expr = None
        if self._FEATURE_MASK.get('custom_levels', False) and args.IsSpecified('custom_level_spec'):
            custom_level_expr = args.custom_level_spec
        return client.Patch(level_ref, description=args.description, title=args.title, basic_level_combine_function=basic_level_combine_function, basic_level_conditions=args.basic_level_spec, custom_level_expr=custom_level_expr)