from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
class BaseSSHHelper(object):
    """Helper class for subcommands that need to connect to instances using SSH.

  Clients can call EnsureSSHKeyIsInProject() to make sure that the
  user's public SSH key is placed in the project metadata before
  proceeding.

  Attributes:
    keys: ssh.Keys, the public/private key pair.
    env: ssh.Environment, the current environment, used by subclasses.
  """
    keys = None

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Please add arguments in alphabetical order except for no- or a clear-
    pair for that argument which can follow the argument itself.
    Args:
      parser: An argparse parser that you can use to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        parser.add_argument('--force-key-file-overwrite', action='store_true', default=None, help='        If enabled, the gcloud command-line tool will regenerate and overwrite\n        the files associated with a broken SSH key without asking for\n        confirmation in both interactive and non-interactive environments.\n\n        If disabled, the files associated with a broken SSH key will not be\n        regenerated and will fail in both interactive and non-interactive\n        environments.')
        parser.add_argument('--ssh-key-file', help="        The path to the SSH key file. By default, this is ``{0}''.\n        ".format(ssh.Keys.DEFAULT_KEY_FILE))

    def Run(self, args):
        """Sets up resources to be used by concrete subclasses.

    Subclasses must call this in their Run() before continuing.

    Args:
      args: argparse.Namespace, arguments that this command was invoked with.

    Raises:
      ssh.CommandNotFoundError: SSH is not supported.
    """
        self.keys = ssh.Keys.FromFilename(args.ssh_key_file)
        self.env = ssh.Environment.Current()
        self.env.RequireSSH()

    def GetInstance(self, client, instance_ref):
        """Fetch an instance based on the given instance_ref."""
        request = (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(instance=instance_ref.Name(), project=instance_ref.project, zone=instance_ref.zone))
        return client.MakeRequests([request])[0]

    def GetProject(self, client, project):
        """Returns the project object.

    Args:
      client: The compute client.
      project: str, the project we are requesting or None for value from
        from properties

    Returns:
      The project object
    """
        return client.MakeRequests([(client.apitools_client.projects, 'Get', client.messages.ComputeProjectsGetRequest(project=project or properties.VALUES.core.project.Get(required=True)))])[0]

    def GetHostKeysFromGuestAttributes(self, client, instance_ref, instance=None, project=None):
        """Get host keys from guest attributes.

    Args:
      client: The compute client.
      instance_ref: The instance object.
      instance: The object representing the instance we are connecting to. If
        either project or instance is None, metadata won't be checked to
        determine if Guest Attributes are enabled.
      project: The object representing the current project. If either project
        or instance is None, metadata won't be checked to determine if
        Guest Attributes are enabled.

    Returns:
      A dictionary of host keys, with the type as the key and the host key
      as the value, or None if Guest Attributes is not enabled.
    """
        if instance and project:
            guest_attributes_enabled = _MetadataHasGuestAttributesEnabled(instance.metadata)
            if guest_attributes_enabled is None:
                project_metadata = project.commonInstanceMetadata
                guest_attributes_enabled = _MetadataHasGuestAttributesEnabled(project_metadata)
            if not guest_attributes_enabled:
                return None
        requests = [(client.apitools_client.instances, 'GetGuestAttributes', client.messages.ComputeInstancesGetGuestAttributesRequest(instance=instance_ref.Name(), project=instance_ref.project, queryPath='hostkeys/', zone=instance_ref.zone))]
        try:
            hostkeys = client.MakeRequests(requests)[0]
        except exceptions.ToolException as e:
            if "The resource 'hostkeys/' of type 'Guest Attribute' was not found." in six.text_type(e):
                hostkeys = None
            else:
                raise e
        hostkey_dict = {}
        if hostkeys is not None:
            for item in hostkeys.queryValue.items:
                if item.namespace == 'hostkeys' and item.key in SUPPORTED_HOSTKEY_TYPES:
                    key_value = item.value.split()[0]
                    try:
                        decoded_key = base64.b64decode(key_value)
                        encoded_key = encoding.Decode(base64.b64encode(decoded_key))
                    except (TypeError, binascii.Error):
                        encoded_key = ''
                    if key_value == encoded_key:
                        hostkey_dict[item.key] = key_value
        return hostkey_dict

    def WriteHostKeysToKnownHosts(self, known_hosts, host_keys, host_key_alias):
        """Writes host keys to known hosts file.

    Only writes keys to known hosts file if there are no existing keys for
    the host.

    Args:
      known_hosts: obj, known_hosts file object.
      host_keys: dict, dictionary of host keys.
      host_key_alias: str, alias for host key entries.
    """
        host_key_entries = []
        for key_type, key in host_keys.items():
            host_key_entry = '{0} {1}'.format(key_type, key)
            host_key_entries.append(host_key_entry)
        host_key_entries.sort()
        new_keys_added = known_hosts.AddMultiple(host_key_alias, host_key_entries, overwrite=False)
        if new_keys_added:
            log.status.Print('Writing {0} keys to {1}'.format(len(host_key_entries), known_hosts.file_path))
        if host_key_entries and (not new_keys_added):
            log.status.Print('Existing host keys found in {0}'.format(known_hosts.file_path))
        known_hosts.Write()

    def _SetProjectMetadata(self, client, new_metadata):
        """Sets the project metadata to the new metadata."""
        errors = []
        client.MakeRequests(requests=[(client.apitools_client.projects, 'SetCommonInstanceMetadata', client.messages.ComputeProjectsSetCommonInstanceMetadataRequest(metadata=new_metadata, project=properties.VALUES.core.project.Get(required=True)))], errors_to_collect=errors)
        if errors:
            utils.RaiseException(errors, SetProjectMetadataError, error_message='Could not add SSH key to project metadata:')

    def SetProjectMetadata(self, client, new_metadata):
        """Sets the project metadata to the new metadata with progress tracker."""
        with progress_tracker.ProgressTracker('Updating project ssh metadata'):
            self._SetProjectMetadata(client, new_metadata)

    def _SetInstanceMetadata(self, client, instance, new_metadata):
        """Sets the instance metadata to the new metadata."""
        errors = []
        zone = instance.zone.split('/')[-1]
        client.MakeRequests(requests=[(client.apitools_client.instances, 'SetMetadata', client.messages.ComputeInstancesSetMetadataRequest(instance=instance.name, metadata=new_metadata, project=properties.VALUES.core.project.Get(required=True), zone=zone))], errors_to_collect=errors)
        if errors:
            utils.RaiseToolException(errors, error_message='Could not add SSH key to instance metadata:')

    def SetInstanceMetadata(self, client, instance, new_metadata):
        """Sets the instance metadata to the new metadata with progress tracker."""
        with progress_tracker.ProgressTracker('Updating instance ssh metadata'):
            self._SetInstanceMetadata(client, instance, new_metadata)

    def EnsureSSHKeyIsInInstance(self, client, user, instance, expiration, legacy=False):
        """Ensures that the user's public SSH key is in the instance metadata.

    Args:
      client: The compute client.
      user: str, the name of the user associated with the SSH key in the
          metadata
      instance: Instance, ensure the SSH key is in the metadata of this instance
      expiration: datetime, If not None, the point after which the key is no
          longer valid.
      legacy: If the key is not present in metadata, add it to the legacy
          metadata entry instead of the default entry.

    Returns:
      bool, True if the key was newly added, False if it was in the metadata
          already
    """
        public_key = self.keys.GetPublicKey()
        new_metadata = _AddSSHKeyToMetadataMessage(client.messages, user, public_key, instance.metadata, expiration=expiration, legacy=legacy)
        has_new_metadata = new_metadata != instance.metadata
        if has_new_metadata:
            self.SetInstanceMetadata(client, instance, new_metadata)
        return has_new_metadata

    def EnsureSSHKeyIsInProject(self, client, user, project=None, expiration=None):
        """Ensures that the user's public SSH key is in the project metadata.

    Args:
      client: The compute client.
      user: str, the name of the user associated with the SSH key in the
          metadata
      project: Project, the project SSH key will be added to
      expiration: datetime, If not None, the point after which the key is no
          longer valid.

    Returns:
      bool, True if the key was newly added, False if it was in the metadata
          already
    """
        public_key = self.keys.GetPublicKey()
        if not project:
            project = self.GetProject(client, None)
        existing_metadata = project.commonInstanceMetadata
        new_metadata = _AddSSHKeyToMetadataMessage(client.messages, user, public_key, existing_metadata, expiration=expiration)
        if new_metadata != existing_metadata:
            self.SetProjectMetadata(client, new_metadata)
            return True
        else:
            return False

    def EnsureSSHKeyExists(self, compute_client, user, instance, project, expiration):
        """Controller for EnsureSSHKey* variants.

    Sends the key to the project metadata or instance metadata,
    and signals whether the key was newly added.

    Args:
      compute_client: The compute client.
      user: str, The user name.
      instance: Instance, the instance to connect to.
      project: Project, the project instance is in.
      expiration: datetime, If not None, the point after which the key is no
          longer valid.


    Returns:
      bool, True if the key was newly added.
    """
        _, ssh_legacy_keys = _GetSSHKeysFromMetadata(instance.metadata)
        if ssh_legacy_keys:
            keys_newly_added = self.EnsureSSHKeyIsInInstance(compute_client, user, instance, expiration, legacy=True)
        elif _MetadataHasBlockProjectSshKeys(instance.metadata):
            keys_newly_added = self.EnsureSSHKeyIsInInstance(compute_client, user, instance, expiration)
        else:
            try:
                keys_newly_added = self.EnsureSSHKeyIsInProject(compute_client, user, project, expiration)
            except SetProjectMetadataError:
                log.info('Could not set project metadata:', exc_info=True)
                log.info('Attempting to set instance metadata.')
                keys_newly_added = self.EnsureSSHKeyIsInInstance(compute_client, user, instance, expiration)
        return keys_newly_added

    def GetConfig(self, host_key_alias, strict_host_key_checking=None, host_keys_to_add=None):
        """Returns a dict of default `ssh-config(5)` options on the OpenSSH format.

    Args:
      host_key_alias: str, Alias of the host key in the known_hosts file.
      strict_host_key_checking: str or None, whether to enforce strict host key
        checking. If None, it will be determined by existence of host_key_alias
        in the known hosts file. Accepted strings are 'yes', 'ask' and 'no'.
      host_keys_to_add: dict, A dictionary of host keys to add to the known
        hosts file.

    Returns:
      Dict with OpenSSH options.
    """
        config = {}
        known_hosts = ssh.KnownHosts.FromDefaultFile()
        config['UserKnownHostsFile'] = known_hosts.file_path
        config['IdentitiesOnly'] = 'yes'
        config['CheckHostIP'] = 'no'
        if not strict_host_key_checking:
            if known_hosts.ContainsAlias(host_key_alias) or host_keys_to_add:
                strict_host_key_checking = 'yes'
            else:
                strict_host_key_checking = 'no'
        if host_keys_to_add:
            self.WriteHostKeysToKnownHosts(known_hosts, host_keys_to_add, host_key_alias)
        config['StrictHostKeyChecking'] = strict_host_key_checking
        config['HostKeyAlias'] = host_key_alias
        config['HashKnownHosts'] = 'no'
        return config