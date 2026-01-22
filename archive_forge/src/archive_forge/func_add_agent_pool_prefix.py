from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def add_agent_pool_prefix(agent_pool_string_or_list):
    """Adds prefix to transfer agent pool(s) if necessary."""
    project_id = properties.VALUES.core.project.Get()
    prefix_to_add = 'projects/{}/agentPools/'.format(project_id)
    result = _add_transfer_prefix(_AGENT_POOLS_PREFIX_REGEX, prefix_to_add, agent_pool_string_or_list)
    if not project_id and result != agent_pool_string_or_list:
        raise ValueError('Project ID not found. Please set a gcloud-wide project, or use full agent pool names (e.g. "projects/[your project ID]/agentPools/[your agent pool name]").')
    return result