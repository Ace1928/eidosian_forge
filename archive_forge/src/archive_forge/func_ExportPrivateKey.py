from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.command_lib.privateca import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def ExportPrivateKey(private_key_output_file, private_key_bytes):
    """Export a private key to a filename, printing a warning to the user.

  Args:
    private_key_output_file: The path of the file to export to.
    private_key_bytes: The content in byte format to export.
  """
    try:
        files.PrivatizeFile(private_key_output_file)
        files.WriteFileContents(private_key_output_file, private_key_bytes)
        os.chmod(private_key_output_file, 256)
        log.warning(KEY_OUTPUT_WARNING.format(private_key_output_file))
    except (files.Error, OSError, IOError):
        raise exceptions.FileOutputError("Error writing to private key output file named '{}'".format(private_key_output_file))