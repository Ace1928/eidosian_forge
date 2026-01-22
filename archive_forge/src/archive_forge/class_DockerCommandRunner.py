import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from getpass import getuser
from shlex import quote
from typing import Dict, List
import click
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.docker import (
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.subprocess_output_util import (
from ray.autoscaler.command_runner import CommandRunnerInterface
class DockerCommandRunner(CommandRunnerInterface):

    def __init__(self, docker_config, **common_args):
        self.ssh_command_runner = SSHCommandRunner(**common_args)
        self.container_name = docker_config['container_name']
        self.docker_config = docker_config
        self.home_dir = None
        self.initialized = False
        use_podman = docker_config.get('use_podman', False)
        self.docker_cmd = 'podman' if use_podman else 'docker'

    def run(self, cmd, timeout=120, exit_on_fail=False, port_forward=None, with_output=False, environment_variables: Dict[str, object]=None, run_env='auto', ssh_options_override_ssh_key='', shutdown_after_run=False):
        if run_env == 'auto':
            run_env = 'host' if not bool(cmd) or cmd.find(self.docker_cmd) == 0 else self.docker_cmd
        if environment_variables:
            cmd = _with_environment_variables(cmd, environment_variables)
        if run_env == 'docker':
            cmd = self._docker_expand_user(cmd, any_char=True)
            if is_using_login_shells():
                cmd = ' '.join(_with_interactive(cmd))
            cmd = with_docker_exec([cmd], container_name=self.container_name, with_interactive=is_using_login_shells(), docker_cmd=self.docker_cmd)[0]
        if shutdown_after_run:
            cmd += '; sudo shutdown -h now'
        return self.ssh_command_runner.run(cmd, timeout=timeout, exit_on_fail=exit_on_fail, port_forward=port_forward, with_output=with_output, ssh_options_override_ssh_key=ssh_options_override_ssh_key)

    def run_rsync_up(self, source, target, options=None):
        options = options or {}
        host_destination = os.path.join(self._get_docker_host_mount_location(self.ssh_command_runner.cluster_name), target.lstrip('/'))
        host_mount_location = os.path.dirname(host_destination.rstrip('/'))
        self.ssh_command_runner.run(f'mkdir -p {host_mount_location} && chown -R {self.ssh_command_runner.ssh_user} {host_mount_location}', silent=is_rsync_silent())
        self.ssh_command_runner.run_rsync_up(source, host_destination, options=options)
        if self._check_container_status() and (not options.get('docker_mount_if_possible', False)):
            if os.path.isdir(source):
                host_destination += '/.'
            prefix = with_docker_exec(['mkdir -p {}'.format(os.path.dirname(self._docker_expand_user(target)))], container_name=self.container_name, with_interactive=is_using_login_shells(), docker_cmd=self.docker_cmd)[0]
            self.ssh_command_runner.run("{} && rsync -e '{} exec -i' -avz {} {}:{}".format(prefix, self.docker_cmd, host_destination, self.container_name, self._docker_expand_user(target)), silent=is_rsync_silent())

    def run_rsync_down(self, source, target, options=None):
        options = options or {}
        host_source = os.path.join(self._get_docker_host_mount_location(self.ssh_command_runner.cluster_name), source.lstrip('/'))
        host_mount_location = os.path.dirname(host_source.rstrip('/'))
        self.ssh_command_runner.run(f'mkdir -p {host_mount_location} && chown -R {self.ssh_command_runner.ssh_user} {host_mount_location}', silent=is_rsync_silent())
        if source[-1] == '/':
            source += '.'
        if not options.get('docker_mount_if_possible', False):
            self.ssh_command_runner.run("rsync -e '{} exec -i' -avz --delete {}:{} {}".format(self.docker_cmd, self.container_name, self._docker_expand_user(source), host_source), silent=is_rsync_silent())
        self.ssh_command_runner.run_rsync_down(host_source, target, options=options)

    def remote_shell_command_str(self):
        inner_str = self.ssh_command_runner.remote_shell_command_str().replace('ssh', 'ssh -tt', 1).strip('\n')
        return inner_str + ' {} exec -it {} /bin/bash\n'.format(self.docker_cmd, self.container_name)

    def _check_docker_installed(self):
        no_exist = 'NoExist'
        output = self.ssh_command_runner.run(f"command -v {self.docker_cmd} || echo '{no_exist}'", with_output=True)
        cleaned_output = output.decode().strip()
        if no_exist in cleaned_output or 'docker' not in cleaned_output:
            if self.docker_cmd == 'docker':
                install_commands = ['curl -fsSL https://get.docker.com -o get-docker.sh', 'sudo sh get-docker.sh', 'sudo usermod -aG docker $USER', 'sudo systemctl restart docker -f']
            else:
                install_commands = ['sudo apt-get update', 'sudo apt-get -y install podman']
            logger.error(f"{self.docker_cmd.capitalize()} not installed. You can install {self.docker_cmd.capitalize()} by adding the following commands to 'initialization_commands':\n" + '\n'.join(install_commands))

    def _check_container_status(self):
        if self.initialized:
            return True
        output = self.ssh_command_runner.run(check_docker_running_cmd(self.container_name, self.docker_cmd), with_output=True).decode('utf-8').strip()
        return 'true' in output.lower() and 'no such object' not in output.lower()

    def _docker_expand_user(self, string, any_char=False):
        user_pos = string.find('~')
        if user_pos > -1:
            if self.home_dir is None:
                self.home_dir = self.ssh_command_runner.run(f'{self.docker_cmd} exec {self.container_name} printenv HOME', with_output=True).decode('utf-8').strip()
            if any_char:
                return string.replace('~/', self.home_dir + '/')
            elif not any_char and user_pos == 0:
                return string.replace('~', self.home_dir, 1)
        return string

    def _check_if_container_restart_is_needed(self, image: str, cleaned_bind_mounts: Dict[str, str]) -> bool:
        re_init_required = False
        running_image = self.run(check_docker_image(self.container_name, self.docker_cmd), with_output=True, run_env='host').decode('utf-8').strip()
        if running_image != image:
            cli_logger.error('A container with name {} is running image {} instead ' + 'of {} (which was provided in the YAML)', self.container_name, running_image, image)
        mounts = self.run(check_bind_mounts_cmd(self.container_name, self.docker_cmd), with_output=True, run_env='host').decode('utf-8').strip()
        try:
            active_mounts = json.loads(mounts)
            active_remote_mounts = {mnt['Destination'].strip('/') for mnt in active_mounts}
            requested_remote_mounts = {self._docker_expand_user(remote).strip('/') for remote in cleaned_bind_mounts.keys()}
            unfulfilled_mounts = requested_remote_mounts - active_remote_mounts
            if unfulfilled_mounts:
                re_init_required = True
                cli_logger.warning('This Docker Container is already running. Restarting the Docker container on this node to pick up the following file_mounts {}', unfulfilled_mounts)
        except json.JSONDecodeError:
            cli_logger.verbose('Unable to check if file_mounts specified in the YAML differ from those on the running container.')
        return re_init_required

    def run_init(self, *, as_head: bool, file_mounts: Dict[str, str], sync_run_yet: bool):
        BOOTSTRAP_MOUNTS = ['~/ray_bootstrap_config.yaml', '~/ray_bootstrap_key.pem']
        specific_image = self.docker_config.get(f'{('head' if as_head else 'worker')}_image', self.docker_config.get('image'))
        self._check_docker_installed()
        if self.docker_config.get('pull_before_run', True):
            assert specific_image, 'Image must be included in config if ' + 'pull_before_run is specified'
            self.run('{} pull {}'.format(self.docker_cmd, specific_image), run_env='host')
        else:
            self.run(f'{self.docker_cmd} image inspect {specific_image} 1> /dev/null  2>&1 || {self.docker_cmd} pull {specific_image}')
        cleaned_bind_mounts = file_mounts.copy()
        for mnt in BOOTSTRAP_MOUNTS:
            cleaned_bind_mounts.pop(mnt, None)
        docker_run_executed = False
        container_running = self._check_container_status()
        requires_re_init = False
        if container_running:
            requires_re_init = self._check_if_container_restart_is_needed(specific_image, cleaned_bind_mounts)
            if requires_re_init:
                self.run(f'{self.docker_cmd} stop {self.container_name}', run_env='host')
        if not container_running or requires_re_init:
            if not sync_run_yet:
                return True
            image_env = self.ssh_command_runner.run(f'{self.docker_cmd} ' + "inspect -f '{{json .Config.Env}}' " + specific_image, with_output=True).decode().strip()
            home_directory = '/root'
            try:
                for env_var in json.loads(image_env):
                    if env_var.startswith('HOME='):
                        home_directory = env_var.split('HOME=')[1]
                        break
            except json.JSONDecodeError as e:
                cli_logger.error(f'Unable to deserialize `image_env` to Python object. The `image_env` is:\n{image_env}')
                raise e
            user_docker_run_options = self.docker_config.get('run_options', []) + self.docker_config.get(f'{('head' if as_head else 'worker')}_run_options', [])
            start_command = docker_start_cmds(self.ssh_command_runner.ssh_user, specific_image, cleaned_bind_mounts, self.container_name, self._configure_runtime(self._auto_configure_shm(user_docker_run_options)), self.ssh_command_runner.cluster_name, home_directory, self.docker_cmd)
            self.run(start_command, run_env='host')
            docker_run_executed = True
        for mount in BOOTSTRAP_MOUNTS:
            if mount in file_mounts:
                if not sync_run_yet:
                    self.run_rsync_up(file_mounts[mount], mount)
                self.ssh_command_runner.run("rsync -e '{cmd} exec -i' -avz {src} {container}:{dst}".format(cmd=self.docker_cmd, src=os.path.join(self._get_docker_host_mount_location(self.ssh_command_runner.cluster_name), mount), container=self.container_name, dst=self._docker_expand_user(mount)))
                try:
                    self.run(f'cat {mount} >/dev/null 2>&1 || sudo chown $(id -u):$(id -g) {mount}')
                except Exception:
                    lsl_string = self.run(f'ls -l {mount}', with_output=True).decode('utf-8').strip()
                    permissions = lsl_string.split(' ')[0]
                    owner = lsl_string.split(' ')[2]
                    group = lsl_string.split(' ')[3]
                    current_user = self.run('whoami', with_output=True).decode('utf-8').strip()
                    cli_logger.warning(f"File ({mount}) is owned by user:{owner} and group:{group} with permissions ({permissions}). The current user ({current_user}) does not have permission to read these files, and Ray may not be able to autoscale. This can be resolved by installing `sudo` in your container, or adding a command like 'chown {current_user} {mount}' to your `setup_commands`.")
        self.initialized = True
        return docker_run_executed

    def _configure_runtime(self, run_options: List[str]) -> List[str]:
        if self.docker_config.get('disable_automatic_runtime_detection'):
            return run_options
        runtime_output = self.ssh_command_runner.run(f'{self.docker_cmd} ' + "info -f '{{.Runtimes}}' ", with_output=True).decode().strip()
        if 'nvidia-container-runtime' in runtime_output:
            try:
                self.ssh_command_runner.run('nvidia-smi', with_output=False)
                return run_options + ['--runtime=nvidia']
            except Exception as e:
                logger.warning('Nvidia Container Runtime is present, but no GPUs found.')
                logger.debug(f'nvidia-smi error: {e}')
                return run_options
        return run_options

    def _auto_configure_shm(self, run_options: List[str]) -> List[str]:
        if self.docker_config.get('disable_shm_size_detection'):
            return run_options
        for run_opt in run_options:
            if '--shm-size' in run_opt:
                logger.info(f'Bypassing automatic SHM-Detection because of `run_option`: {run_opt}')
                return run_options
        try:
            shm_output = self.ssh_command_runner.run('cat /proc/meminfo || true', with_output=True).decode().strip()
            available_memory = int([ln for ln in shm_output.split('\n') if 'MemAvailable' in ln][0].split()[1])
            available_memory_bytes = available_memory * 1024
            shm_size = min(available_memory_bytes * DEFAULT_OBJECT_STORE_MEMORY_PROPORTION * 1.1, DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES)
            return run_options + [f"--shm-size='{shm_size}b'"]
        except Exception as e:
            logger.warning(f'Received error while trying to auto-compute SHM size {e}')
            return run_options

    def _get_docker_host_mount_location(self, cluster_name: str) -> str:
        """Return the docker host mount directory location."""
        from ray.autoscaler.sdk import get_docker_host_mount_location
        return get_docker_host_mount_location(cluster_name)