from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
import sys
import time
from apitools.base.py import list_pager
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import times
import six
class SSH(object):
    """Helper class to SSH to the VM associated with the TPU node."""

    def __init__(self, release_track):
        holder = base_classes.ComputeApiHolder(release_track)
        self.release_track = release_track
        self.client = holder.client
        self.resources = holder.resources

    def _DefaultArgsForSSH(self, args):
        args.plain = None
        args.strict_host_key_checking = 'no'
        args.force_key_file_overwrite = None
        args.ssh_key_file = None
        return args

    def _GetHostKeyFromInstance(self, zone, ssh_helper, instance):
        """Wrapper around SSH Utils to get the host keys for SSH."""
        instance_ref = instance_flags.SSH_INSTANCE_RESOLVER.ResolveResources([instance.name], compute_scope.ScopeEnum.ZONE, zone, self.resources, scope_lister=instance_flags.GetInstanceZoneScopeLister(self.client))[0]
        project = ssh_helper.GetProject(self.client, instance_ref.project)
        host_keys = ssh_helper.GetHostKeysFromGuestAttributes(self.client, instance_ref, instance, project)
        if host_keys is not None and (not host_keys):
            log.status.Print('Unable to retrieve host keys from instance metadata. Continuing.')
        return host_keys

    def _GetSSHOptions(self, name, ssh_helper, instance, host_keys):
        options = ssh_helper.GetConfig(ssh_utils.HostKeyAlias(instance), strict_host_key_checking='no', host_keys_to_add=host_keys)
        os.environ['TPU_NAME'] = name
        options['SendEnv'] = 'TPU_NAME'
        return options

    def _WaitForSSHKeysToPropagate(self, ssh_helper, remote, identity_file, user, instance, options, putty_force_connect=False):
        """Waits for SSH keys to propagate in order to SSH to the instance."""
        ssh_helper.EnsureSSHKeyExists(self.client, user, instance, ssh_helper.GetProject(self.client, properties.VALUES.core.project.Get(required=True)), times.Now() + datetime.timedelta(seconds=300))
        ssh_poller = ssh.SSHPoller(remote=remote, identity_file=identity_file, options=options, max_wait_ms=300 * 1000)
        try:
            ssh_poller.Poll(ssh_helper.env, putty_force_connect=putty_force_connect)
        except retry.WaitException:
            raise ssh_utils.NetworkError()

    def SSHToInstance(self, args, instance):
        """Helper to manage authentication followed by SSH to the instance."""
        args = self._DefaultArgsForSSH(args)
        external_nat = ssh_utils.GetExternalIPAddress(instance)
        log.status.Print('Trying to SSH to VM with NAT IP:{}'.format(external_nat))
        args.ssh_key_file = ssh.Keys.DEFAULT_KEY_FILE
        ssh_helper = ssh_utils.BaseSSHCLIHelper()
        ssh_helper.Run(args)
        identity_file = ssh_helper.keys.key_file
        user, _ = ssh_utils.GetUserAndInstance(args.name)
        host_keys = self._GetHostKeyFromInstance(args.zone, ssh_helper, instance)
        options = self._GetSSHOptions(args.name, ssh_helper, instance, host_keys)
        public_key = ssh_helper.keys.GetPublicKey().ToEntry(include_comment=True)
        oslogin_state = ssh.GetOsloginState(instance, ssh_helper.GetProject(self.client, properties.VALUES.core.project.Get(required=True)), user, public_key, None, self.release_track, username_requested=False, messages=self.client.messages)
        user = oslogin_state.user
        remote = ssh.Remote(external_nat, user)
        putty_force_connect = not oslogin_state.oslogin_2fa_enabled and properties.VALUES.ssh.putty_force_connect.GetBool()
        if not oslogin_state.oslogin_enabled:
            self._WaitForSSHKeysToPropagate(ssh_helper, remote, identity_file, user, instance, options, putty_force_connect)
        extra_flags = []
        if args.forward_ports:
            extra_flags.extend(['-A', '-L', '6006:localhost:6006', '-L', '8888:localhost:8888'])
        ssh_cmd_args = {'remote': remote, 'identity_file': identity_file, 'options': options, 'extra_flags': extra_flags}
        cmd = ssh.SSHCommand(**ssh_cmd_args)
        max_attempts = 10
        sleep_interval = 30
        for i in range(max_attempts):
            try:
                log.status.Print('SSH Attempt #{}...'.format(i))
                return_code = cmd.Run(ssh_helper.env, putty_force_connect=putty_force_connect)
                if return_code:
                    sys.exit(return_code)
            except ssh.CommandError as e:
                if i == max_attempts - 1:
                    raise e
                log.status.Print('Retrying: SSH command error: {}'.format(six.text_type(e)))
                time.sleep(sleep_interval)
                continue
            break