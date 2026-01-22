from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import openssl_encryption_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as core_encoding
from googlecloudsdk.core.util import files
class ResetWindowsPassword(base.UpdateCommand):
    """Reset and return a password for a Windows machine instance.

  *{command}* allows a user to reset and retrieve a password for
  a Windows virtual machine instance. If the Windows account does not
  exist, this command will cause the account to be created and the
  password for that new account will be returned.

  For Windows instances that are running a domain controller, running
  this command creates a new domain user if the user does not exist,
  or resets the password if the user does exist. It is not possible to
  use this command to create a local user on a domain-controller
  instance.

  NOTE: When resetting passwords or adding a new user to a Domain Controller
  in this way, the user will be added to the built in Admin Group on the
  Domain Controller. This will give the user wide reaching permissions. If
  this is not the desired outcome then Active Directory Users and Computers
  should be used instead.

  For all other instances, including domain-joined instances, running
  this command creates a local user or resets the password for a local
  user.

  WARNING: Resetting a password for an existing user can cause the
  loss of data encrypted with the current Windows password, such as
  encrypted files or stored passwords.

  The user running this command must have write permission for the
  Google Compute Engine project containing the Windows instance.

  ## EXAMPLES

  To reset the password for user 'foo' on a Windows instance 'my-instance' in
  zone 'us-central1-f', run:

    $ {command} my-instance --zone=us-central1-f --user=foo
  """
    category = base.TOOLS_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('[private]text')
        parser.add_argument('--user', help="        ``USER'' specifies the username to get the password for.\n        If omitted, the username is derived from your authenticated\n        account email address.\n        ")
        instance_flags.INSTANCE_ARG.AddArgument(parser)

    def GetGetRequest(self, client, instance_ref):
        return (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))

    def GetSetRequest(self, client, instance_ref, replacement):
        return (client.apitools_client.instances, 'SetMetadata', client.messages.ComputeInstancesSetMetadataRequest(metadata=replacement.metadata, **instance_ref.AsDict()))

    def CreateReference(self, client, resources, args):
        return instance_flags.INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=instance_flags.GetInstanceZoneScopeLister(client))

    def Modify(self, client, existing):
        new_object = encoding.CopyProtoMessage(existing)
        existing_metadata = getattr(existing, 'metadata', None)
        new_metadata = metadata_utils.ConstructMetadataMessage(message_classes=client.messages, metadata={METADATA_KEY: self._UpdateWindowsKeysValue(existing_metadata)}, existing_metadata=existing_metadata)
        new_object.metadata = new_metadata
        return new_object

    def _ConstructWindowsKeyEntry(self, user, modulus, exponent, email):
        """Return a JSON formatted entry for 'windows-keys'."""
        expire_str = time_util.CalculateExpiration(RSA_KEY_EXPIRATION_TIME_SEC)
        windows_key_data = {'userName': user, 'modulus': core_encoding.Decode(modulus), 'exponent': core_encoding.Decode(exponent), 'email': email, 'expireOn': expire_str}
        windows_key_entry = json.dumps(windows_key_data, sort_keys=True)
        return windows_key_entry

    def _UpdateWindowsKeysValue(self, existing_metadata):
        """Returns a string appropriate for the metadata.

    Values are removed if they have expired and non-expired keys are removed
    from the head of the list only if the total key size is greater than
    MAX_METADATA_VALUE_SIZE_IN_BYTES.

    Args:
      existing_metadata: The existing metadata for the instance to be updated.

    Returns:
      A new-line-joined string of Windows keys.
    """
        windows_keys = []
        self.old_metadata_keys = []
        for item in existing_metadata.items:
            if item.key == METADATA_KEY:
                windows_keys = [key.strip() for key in item.value.split('\n') if key]
            if item.key in OLD_METADATA_KEYS:
                self.old_metadata_keys.append(item.key)
        windows_keys.append(self.windows_key_entry)
        keys = []
        bytes_consumed = 0
        for key in reversed(windows_keys):
            num_bytes = len(key + '\n')
            key_expired = False
            try:
                key_data = json.loads(key)
                if time_util.IsExpired(key_data['expireOn']):
                    key_expired = True
            except (ValueError, KeyError):
                pass
            if key_expired:
                log.debug('The following Windows key has expired and will be removed from your project: {0}'.format(key))
            elif bytes_consumed + num_bytes > constants.MAX_METADATA_VALUE_SIZE_IN_BYTES:
                log.debug('The following Windows key will be removed from your project because your windows keys metadata value has reached its maximum allowed size of {0} bytes: {1}'.format(constants.MAX_METADATA_VALUE_SIZE_IN_BYTES, key))
            else:
                keys.append(key)
                bytes_consumed += num_bytes
        log.debug('Number of Windows Keys: {0}'.format(len(keys)))
        keys.reverse()
        return '\n'.join(keys)

    def _GetSerialPortOutput(self, client, instance_ref, port=4):
        """Returns the serial port output for self.instance_ref."""
        request = (client.apitools_client.instances, 'GetSerialPortOutput', client.messages.ComputeInstancesGetSerialPortOutputRequest(port=port, **instance_ref.AsDict()))
        objects = client.MakeRequests([request])
        return objects[0].contents

    def _GetEncryptedPasswordFromSerialPort(self, client, instance_ref, search_modulus):
        """Returns the decrypted password from the data in the serial port."""
        encrypted_password_data = {}
        start_time = time_util.CurrentTimeSec()
        count = 1
        agent_ready = False
        while not encrypted_password_data:
            log.debug('Get Serial Port Output, Try {0}'.format(count))
            if time_util.CurrentTimeSec() > start_time + WINDOWS_PASSWORD_TIMEOUT_SEC:
                raise utils.TimeoutError(TIMEOUT_ERROR.format(time_util.CurrentDatetimeUtc()))
            serial_port_output = self._GetSerialPortOutput(client, instance_ref, port=4).split('\n')
            for line in reversed(serial_port_output):
                try:
                    encrypted_password_dict = json.loads(line)
                except ValueError:
                    continue
                modulus = encrypted_password_dict.get('modulus')
                if modulus or encrypted_password_dict.get('ready'):
                    agent_ready = True
                if not encrypted_password_dict.get('encryptedPassword'):
                    continue
                if core_encoding.Decode(search_modulus) == core_encoding.Decode(modulus):
                    encrypted_password_data = encrypted_password_dict
                    break
            if not agent_ready:
                if self.old_metadata_keys:
                    message = OLD_WINDOWS_BUILD_ERROR.format(instance_ref.instance, instance_ref.zone)
                    raise utils.WrongInstanceTypeError(message)
                else:
                    message = NOT_READY_ERROR
                    raise utils.InstanceNotReadyError(message)
            time_util.Sleep(POLLING_SEC)
            count += 1
        encrypted_password = encrypted_password_data['encryptedPassword']
        return encrypted_password

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        start = time_util.CurrentTimeSec()
        openssl_executable = files.FindExecutableOnPath('openssl')
        if windows_encryption_utils:
            crypt = windows_encryption_utils.WinCrypt()
        elif openssl_executable:
            crypt = openssl_encryption_utils.OpensslCrypt(openssl_executable)
        else:
            raise utils.MissingDependencyError('Your platform does not support OpenSSL.')
        email = properties.VALUES.core.account.GetOrFail()
        if args.user:
            user = args.user
        else:
            user = gaia.MapGaiaEmailToDefaultAccountName(email)
        if args.instance_name == user:
            raise utils.InvalidUserError(MACHINE_USERNAME_SAME_ERROR.format(user, args.instance_name))
        message = RESET_PASSWORD_WARNING.format(user)
        prompt_string = 'Would you like to set or reset the password for [{0}]'.format(user)
        console_io.PromptContinue(message=message, prompt_string=prompt_string, cancel_on_no=True)
        log.status.Print('Resetting and retrieving password for [{0}] on [{1}]'.format(user, args.instance_name))
        key = crypt.GetKeyPair()
        modulus, exponent = crypt.GetModulusExponentFromPublicKey(crypt.GetPublicKey(key))
        self.windows_key_entry = self._ConstructWindowsKeyEntry(user, modulus, exponent, email)
        instance_ref = self.CreateReference(client, holder.resources, args)
        get_request = self.GetGetRequest(client, instance_ref)
        objects = client.MakeRequests([get_request])
        new_object = self.Modify(client, objects[0])
        if objects[0] == new_object:
            log.status.Print('No change requested; skipping update for [{0}].'.format(objects[0].name))
            return objects
        updated_instance = client.MakeRequests([self.GetSetRequest(client, instance_ref, new_object)])[0]
        enc_password = self._GetEncryptedPasswordFromSerialPort(client, instance_ref, modulus)
        password = crypt.DecryptMessage(key, enc_password)
        try:
            access_configs = updated_instance.networkInterfaces[0].accessConfigs
            external_ip_address = access_configs[0].natIP
        except (KeyError, IndexError) as _:
            log.warning(NO_IP_WARNING.format(updated_instance.name))
            external_ip_address = None
        if self.old_metadata_keys:
            log.warning(OLD_KEYS_WARNING.format(instance_ref.instance, instance_ref.instance, instance_ref.zone, ','.join(self.old_metadata_keys)))
        log.info('Total Elapsed Time: {0}'.format(time_util.CurrentTimeSec() - start))
        connection_info = {'username': user, 'password': password, 'ip_address': external_ip_address}
        return connection_info