from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ProcessLevels(ref, args, req):
    """Hook to format levels and validate all policies."""
    del ref
    policies_to_check = {}
    param = {}
    policy_ref = None
    if args.IsKnownAndSpecified('policy'):
        try:
            policy_ref = resources.REGISTRY.Parse(args.GetValue('policy'), collection='accesscontextmanager.accessPolicies')
        except:
            raise calliope_exceptions.InvalidArgumentException('--policy', 'The input must be the full identifier for the access policy, such as `123` or `accessPolicies/123.')
        param = {'accessPoliciesId': policy_ref.Name()}
        policies_to_check['--policy'] = policy_ref.RelativeName()
    level_refs = _ParseLevelRefs(req, param, is_dry_run=False) if args.IsKnownAndSpecified('level') else []
    dry_run_level_refs = _ParseLevelRefs(req, param, is_dry_run=True) if args.IsKnownAndSpecified('dry_run_level') else []
    level_parents = [x.Parent() for x in level_refs]
    dry_run_level_parents = [x.Parent() for x in dry_run_level_refs]
    if not all((x == level_parents[0] for x in level_parents)):
        raise ConflictPolicyException(['--level'])
    if not all((x == dry_run_level_parents[0] for x in dry_run_level_parents)):
        raise ConflictPolicyException(['--dry-run-level'])
    if level_parents:
        policies_to_check['--level'] = level_parents[0].RelativeName()
    if dry_run_level_parents:
        policies_to_check['--dry-run-level'] = dry_run_level_parents[0].RelativeName()
    flags_to_complain = list(policies_to_check.keys())
    flags_to_complain.sort()
    policies_values = list(policies_to_check.values())
    if not all((x == policies_values[0] for x in policies_values)):
        raise ConflictPolicyException(flags_to_complain)
    if level_refs:
        req.gcpUserAccessBinding.accessLevels = [x.RelativeName() for x in level_refs]
    if dry_run_level_refs:
        req.gcpUserAccessBinding.dryRunAccessLevels = [x.RelativeName() for x in dry_run_level_refs]
    return req