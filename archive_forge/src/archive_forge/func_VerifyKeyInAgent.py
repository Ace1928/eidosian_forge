from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import sys
import threading
import time
from apitools.base.py import encoding_helper
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import exceptions as tpu_exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util.files import FileWriter
import six
def VerifyKeyInAgent(identity_file):
    """Verifies that the ssh-agent holds the SSH key."""
    cmd = ['ssh-keygen', '-lf', identity_file]
    keygen_out = io.StringIO()
    err = io.StringIO()
    retcode = execution_utils.Exec(cmd, no_exit=True, out_func=keygen_out.write, err_func=err.write)
    if retcode != 0:
        log.debug('ssh-keygen exited with error {}'.format(err.getvalue()))
        log.warning('Cannot generate fingerprint of SSH key. Command may stall.')
        return
    fingerprint_entry = keygen_out.getvalue()
    if len(fingerprint_entry.split()) <= 1:
        log.debug('ssh-keygen returned fingerprint entry in invalid format: "{}"'.format(fingerprint_entry))
        return
    fingerprint = fingerprint_entry.split()[1]
    cmd = ['ssh-add', '-l']
    out = io.StringIO()
    retcode = execution_utils.Exec(cmd, no_exit=True, out_func=out.write, err_func=err.write)
    if retcode != 0:
        log.debug('ssh-add exited with error {}'.format(err.getvalue()))
        log.warning('Cannot retrieve keys in ssh-agent. Command may stall.')
        return
    if fingerprint not in out.getvalue():
        raise tpu_exceptions.SSHKeyNotInAgent(identity_file)