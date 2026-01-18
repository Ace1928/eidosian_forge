from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.netvisor.netvisor import run_commands

    This method executes the cli command on the target node(s) and returns the
    output. The module then exits based on the output.
    :param cli: the complete cli string to be executed on the target node(s).
    :param state_map: Provides state of the command.
    :param module: The Ansible module to fetch command
    