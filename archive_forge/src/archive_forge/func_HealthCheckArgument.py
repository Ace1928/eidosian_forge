from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def HealthCheckArgument(protocol_string, name=None, required=True, plural=False, include_regional_health_check=True, scope_flags_usage=compute_flags.ScopeFlagsUsage.GENERATE_DEDICATED_SCOPE_FLAGS):
    return compute_flags.ResourceArgument(name=name, resource_name='{} health check'.format(protocol_string), completer=compute_completers.HealthChecksCompleter, plural=plural, required=required, scope_flags_usage=scope_flags_usage, global_collection='compute.healthChecks', regional_collection='compute.regionHealthChecks' if include_regional_health_check else None, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION if include_regional_health_check else None)