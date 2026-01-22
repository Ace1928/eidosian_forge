import os
import subprocess
from typing import Dict, List, Tuple
from ray.autoscaler._private.docker import with_docker_exec
from ray.autoscaler.command_runner import CommandRunnerInterface
def _docker_expand_user(self, string, any_char=False):
    user_pos = string.find('~')
    if user_pos > -1:
        if self.home_dir is None:
            self.home_dir = self._run_shell(with_docker_exec(['printenv HOME'], container_name=self.container_name, docker_cmd=self.docker_cmd)).strip()
        if any_char:
            return string.replace('~/', self.home_dir + '/')
        elif not any_char and user_pos == 0:
            return string.replace('~', self.home_dir, 1)
    return string