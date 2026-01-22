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
class SSHCommandRunner(CommandRunnerInterface):

    def __init__(self, log_prefix, node_id, provider, auth_config, cluster_name, process_runner, use_internal_ip):
        ssh_control_hash = hashlib.md5(cluster_name.encode()).hexdigest()
        ssh_user_hash = hashlib.md5(getuser().encode()).hexdigest()
        ssh_control_path = '/tmp/ray_ssh_{}/{}'.format(ssh_user_hash[:HASH_MAX_LENGTH], ssh_control_hash[:HASH_MAX_LENGTH])
        self.cluster_name = cluster_name
        self.log_prefix = log_prefix
        self.process_runner = process_runner
        self.node_id = node_id
        self.use_internal_ip = use_internal_ip
        self.provider = provider
        self.ssh_private_key = auth_config.get('ssh_private_key')
        self.ssh_user = auth_config['ssh_user']
        self.ssh_control_path = ssh_control_path
        self.ssh_ip = None
        self.ssh_proxy_command = auth_config.get('ssh_proxy_command', None)
        self.ssh_options = SSHOptions(self.ssh_private_key, self.ssh_control_path, ProxyCommand=self.ssh_proxy_command)

    def _get_node_ip(self):
        if self.use_internal_ip:
            return self.provider.internal_ip(self.node_id)
        else:
            return self.provider.external_ip(self.node_id)

    def _wait_for_ip(self, deadline):
        ip = self._get_node_ip()
        if ip is not None:
            cli_logger.labeled_value('Fetched IP', ip)
            return ip
        interval = AUTOSCALER_NODE_SSH_INTERVAL_S
        with cli_logger.group('Waiting for IP'):
            while time.time() < deadline and (not self.provider.is_terminated(self.node_id)):
                ip = self._get_node_ip()
                if ip is not None:
                    cli_logger.labeled_value('Received', ip)
                    return ip
                cli_logger.print('Not yet available, retrying in {} seconds', cf.bold(str(interval)))
                time.sleep(interval)
        return None

    def _set_ssh_ip_if_required(self):
        if self.ssh_ip is not None:
            return
        deadline = time.time() + AUTOSCALER_NODE_START_WAIT_S
        with LogTimer(self.log_prefix + 'Got IP'):
            ip = self._wait_for_ip(deadline)
            cli_logger.doassert(ip is not None, 'Could not get node IP.')
            assert ip is not None, 'Unable to find IP of node'
        self.ssh_ip = ip
        try:
            os.makedirs(self.ssh_control_path, mode=448, exist_ok=True)
        except OSError as e:
            cli_logger.warning('{}', str(e))

    def _run_helper(self, final_cmd, with_output=False, exit_on_fail=False, silent=False):
        """Run a command that was already setup with SSH and `bash` settings.

        Args:
            cmd (List[str]):
                Full command to run. Should include SSH options and other
                processing that we do.
            with_output (bool):
                If `with_output` is `True`, command stdout will be captured and
                returned.
            exit_on_fail (bool):
                If `exit_on_fail` is `True`, the process will exit
                if the command fails (exits with a code other than 0).

        Raises:
            ProcessRunnerError if using new log style and disabled
                login shells.
            click.ClickException if using login shells.
        """
        try:
            if not with_output:
                return run_cmd_redirected(final_cmd, process_runner=self.process_runner, silent=silent, use_login_shells=is_using_login_shells())
            else:
                return self.process_runner.check_output(final_cmd)
        except subprocess.CalledProcessError as e:
            joined_cmd = ' '.join(final_cmd)
            if not is_using_login_shells():
                raise ProcessRunnerError('Command failed', 'ssh_command_failed', code=e.returncode, command=joined_cmd)
            if exit_on_fail:
                raise click.ClickException('Command failed:\n\n  {}\n'.format(joined_cmd)) from None
            else:
                fail_msg = 'SSH command failed.'
                if is_output_redirected():
                    fail_msg += ' See above for the output from the failure.'
                raise click.ClickException(fail_msg) from None
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    def run(self, cmd, timeout=120, exit_on_fail=False, port_forward=None, with_output=False, environment_variables: Dict[str, object]=None, run_env='auto', ssh_options_override_ssh_key='', shutdown_after_run=False, silent=False):
        if shutdown_after_run:
            cmd += '; sudo shutdown -h now'
        if ssh_options_override_ssh_key:
            if self.ssh_proxy_command:
                ssh_options = SSHOptions(ssh_options_override_ssh_key, ProxyCommand=self.ssh_proxy_command)
            else:
                ssh_options = SSHOptions(ssh_options_override_ssh_key)
        else:
            ssh_options = self.ssh_options
        assert isinstance(ssh_options, SSHOptions), 'ssh_options must be of type SSHOptions, got {}'.format(type(ssh_options))
        self._set_ssh_ip_if_required()
        if is_using_login_shells():
            ssh = ['ssh', '-tt']
        else:
            ssh = ['ssh']
        if port_forward:
            with cli_logger.group('Forwarding ports'):
                if not isinstance(port_forward, list):
                    port_forward = [port_forward]
                for local, remote in port_forward:
                    cli_logger.verbose('Forwarding port {} to port {} on localhost.', cf.bold(local), cf.bold(remote))
                    ssh += ['-L', '{}:localhost:{}'.format(remote, local)]
        final_cmd = ssh + ssh_options.to_ssh_options_list(timeout=timeout) + ['{}@{}'.format(self.ssh_user, self.ssh_ip)]
        if cmd:
            if environment_variables:
                cmd = _with_environment_variables(cmd, environment_variables)
            if is_using_login_shells():
                final_cmd += _with_interactive(cmd)
            else:
                final_cmd += [cmd]
        else:
            final_cmd.append('while true; do sleep 86400; done')
        cli_logger.verbose('Running `{}`', cf.bold(cmd))
        with cli_logger.indented():
            cli_logger.very_verbose('Full command is `{}`', cf.bold(' '.join(final_cmd)))
        if cli_logger.verbosity > 0:
            with cli_logger.indented():
                return self._run_helper(final_cmd, with_output, exit_on_fail, silent=silent)
        else:
            return self._run_helper(final_cmd, with_output, exit_on_fail, silent=silent)

    def _create_rsync_filter_args(self, options):
        rsync_excludes = options.get('rsync_exclude') or []
        rsync_filters = options.get('rsync_filter') or []
        exclude_args = [['--exclude', rsync_exclude] for rsync_exclude in rsync_excludes]
        filter_args = [['--filter', 'dir-merge,- {}'.format(rsync_filter)] for rsync_filter in rsync_filters]
        return [arg for args_list in exclude_args + filter_args for arg in args_list]

    def run_rsync_up(self, source, target, options=None):
        self._set_ssh_ip_if_required()
        options = options or {}
        command = ['rsync']
        command += ['--rsh', subprocess.list2cmdline(['ssh'] + self.ssh_options.to_ssh_options_list(timeout=120))]
        command += ['-avz']
        command += self._create_rsync_filter_args(options=options)
        command += [source, '{}@{}:{}'.format(self.ssh_user, self.ssh_ip, target)]
        cli_logger.verbose('Running `{}`', cf.bold(' '.join(command)))
        self._run_helper(command, silent=is_rsync_silent())

    def run_rsync_down(self, source, target, options=None):
        self._set_ssh_ip_if_required()
        command = ['rsync']
        command += ['--rsh', subprocess.list2cmdline(['ssh'] + self.ssh_options.to_ssh_options_list(timeout=120))]
        command += ['-avz']
        command += self._create_rsync_filter_args(options=options)
        command += ['{}@{}:{}'.format(self.ssh_user, self.ssh_ip, source), target]
        cli_logger.verbose('Running `{}`', cf.bold(' '.join(command)))
        self._run_helper(command, silent=is_rsync_silent())

    def remote_shell_command_str(self):
        if self.ssh_private_key:
            return 'ssh -o IdentitiesOnly=yes -i {} {}@{}\n'.format(self.ssh_private_key, self.ssh_user, self.ssh_ip)
        else:
            return 'ssh -o IdentitiesOnly=yes {}@{}\n'.format(self.ssh_user, self.ssh_ip)