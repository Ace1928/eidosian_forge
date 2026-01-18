from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.command_lib.privateca import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Export a private key to a filename, printing a warning to the user.

  Args:
    private_key_output_file: The path of the file to export to.
    private_key_bytes: The content in byte format to export.
  