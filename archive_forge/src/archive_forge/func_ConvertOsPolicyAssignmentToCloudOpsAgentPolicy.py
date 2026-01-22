import json
from googlecloudsdk.api_lib.compute.instances.ops_agents import cloud_ops_agents_policy as agents_policy
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as osconfig
def ConvertOsPolicyAssignmentToCloudOpsAgentPolicy(os_policy_assignment: osconfig.OSPolicyAssignment) -> agents_policy.OpsAgentsPolicy:
    """Converts OS Config guest policy to Ops Agent policy."""
    instance_filter = os_policy_assignment.instanceFilter
    if len(os_policy_assignment.osPolicies) > 1:
        raise ValueError('Multiple OS Policies found.')
    description = os_policy_assignment.osPolicies[0].description
    try:
        agents_rule_str = description.split(' | ', maxsplit=1)[1]
        agents_rule = json.loads(agents_rule_str)
    except (IndexError, ValueError, AttributeError) as e:
        raise ValueError('Malformed OS Policy: %s' % description) from e
    return agents_policy.OpsAgentsPolicy(agents_rule=agents_rule, instance_filter=instance_filter)