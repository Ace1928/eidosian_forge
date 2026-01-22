from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
def PrepareEnvironment(args):
    """Ensures that the user's environment is ready to accept SSH connections."""
    client = apis.GetClientInstance('cloudshell', 'v1')
    messages = apis.GetMessagesModule('cloudshell', 'v1')
    operations_client = apis.GetClientInstance('cloudshell', 'v1')
    ssh_env = ssh.Environment.Current()
    ssh_env.RequireSSH()
    keys = ssh.Keys.FromFilename(filename=args.ssh_key_file)
    keys.EnsureKeysExist(overwrite=args.force_key_file_overwrite)
    environment = client.users_environments.Get(messages.CloudshellUsersEnvironmentsGetRequest(name=DEFAULT_ENVIRONMENT_NAME))
    key = keys.GetPublicKey().ToEntry()
    has_key = False
    for candidate in environment.publicKeys:
        if key == candidate:
            has_key = True
            break
    if not has_key:
        add_public_key_operation = client.users_environments.AddPublicKey(messages.CloudshellUsersEnvironmentsAddPublicKeyRequest(environment=DEFAULT_ENVIRONMENT_NAME, addPublicKeyRequest=messages.AddPublicKeyRequest(key=key)))
        environment = waiter.WaitFor(EnvironmentPoller(client.users_environments, operations_client.operations), add_public_key_operation, 'Pushing your public key to Cloud Shell', sleep_ms=500, max_wait_ms=None)
    if environment.state != messages.Environment.StateValueValuesEnum.RUNNING:
        log.Print('Starting your Cloud Shell machine...')
        access_token = None
        if args.IsKnownAndSpecified('authorize_session') and args.authorize_session:
            access_token = store.GetFreshAccessTokenIfEnabled(min_expiry_duration=MIN_CREDS_EXPIRY_SECONDS)
        start_operation = client.users_environments.Start(messages.CloudshellUsersEnvironmentsStartRequest(name=DEFAULT_ENVIRONMENT_NAME, startEnvironmentRequest=messages.StartEnvironmentRequest(accessToken=access_token)))
        environment = waiter.WaitFor(EnvironmentPoller(client.users_environments, operations_client.operations), start_operation, 'Waiting for your Cloud Shell machine to start', sleep_ms=500, max_wait_ms=None)
    if not environment.sshHost:
        raise core_exceptions.Error('The Cloud Shell machine did not start.')
    return ConnectionInfo(ssh_env=ssh_env, user=environment.sshUsername, host=environment.sshHost, port=environment.sshPort, key=keys.key_file)