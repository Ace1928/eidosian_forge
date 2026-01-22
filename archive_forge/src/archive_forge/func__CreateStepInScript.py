from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateStepInScript(messages, agent_rule, os_type):
    """Create scriptRun step in guest policy recipe section.

  Args:
    messages: os config guest policy api messages.
    agent_rule: logging or metrics agent rule.
    os_type: it contains os_version, os_short_name.

  Returns:
    Step of script to be run in Recipe section. If the package state is
    "removed", this run script is empty. We still keep the software recipe to
    maintain versioning of the software recipe as the policy gets updated.
  """
    step = messages.SoftwareRecipeStep()
    step.scriptRun = messages.SoftwareRecipeStepRunScript()
    agent_version = '' if agent_rule.version == 'latest' else agent_rule.version
    if os_type.short_name in _YUM_OS:
        clear_prev_repo = _AGENT_RULE_TEMPLATES[agent_rule.type].yum_package.clear_prev_repo
        install_with_version = _AGENT_RULE_TEMPLATES[agent_rule.type].install_with_version % agent_version
    if os_type.short_name in _APT_OS:
        clear_prev_repo = _AGENT_RULE_TEMPLATES[agent_rule.type].apt_package.clear_prev_repo
        install_with_version = _AGENT_RULE_TEMPLATES[agent_rule.type].install_with_version % agent_version
    if os_type.short_name in _SUSE_OS:
        clear_prev_repo = _AGENT_RULE_TEMPLATES[agent_rule.type].zypper_package.clear_prev_repo
        install_with_version = _AGENT_RULE_TEMPLATES[agent_rule.type].install_with_version % agent_version
    if os_type.short_name in _WINDOWS_OS:
        if agent_rule.version == 'latest' or '*.*' in agent_rule.version:
            agent_version = ''
        else:
            agent_version = '.x86_64.%s@1' % agent_rule.version
    if agent_rule.package_state == agent_policy.OpsAgentPolicy.AgentRule.PackageState.REMOVED:
        step.scriptRun.script = _EMPTY_SOFTWARE_RECIPE_SCRIPT
    elif os_type.short_name in _WINDOWS_OS:
        step.scriptRun.interpreter = messages.SoftwareRecipeStepRunScript.InterpreterValueValuesEnum.POWERSHELL
        step.scriptRun.script = _AGENT_RULE_TEMPLATES[agent_rule.type].win_run_agent % agent_version
    else:
        step.scriptRun.script = _AGENT_RULE_TEMPLATES[agent_rule.type].run_agent % {'install': install_with_version, 'clear_prev_repo': clear_prev_repo}
    return step