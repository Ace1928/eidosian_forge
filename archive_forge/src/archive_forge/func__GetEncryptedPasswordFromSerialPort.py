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