from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateYumPkgRepos(messages, repo_distro, agent_rules):
    yum_pkg_repos = []
    for agent_rule in agent_rules:
        template = _AGENT_RULE_TEMPLATES[agent_rule.type]
        repo_name = template.yum_package.repo % (repo_distro, _GetRepoSuffix(agent_rule.version))
        yum_pkg_repos.append(_CreateYumPkgRepo(messages, template.repo_id, template.display_name, repo_name))
    return yum_pkg_repos