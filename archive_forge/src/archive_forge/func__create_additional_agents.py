from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import socket
import subprocess
import sys
from googlecloudsdk.api_lib.transfer import agent_pools_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from oauth2client import client as oauth2_client
def _create_additional_agents(agent_count, agent_id_prefix, docker_command):
    """Creates multiple identical agents."""
    for i in range(1, agent_count):
        if agent_id_prefix:
            docker_command_to_run = docker_command[:-1] + ['--agent-id-prefix={}{}'.format(agent_id_prefix, i)]
        else:
            docker_command_to_run = docker_command
        subprocess.run(docker_command_to_run, check=True)
        _log_created_agent(docker_command_to_run)