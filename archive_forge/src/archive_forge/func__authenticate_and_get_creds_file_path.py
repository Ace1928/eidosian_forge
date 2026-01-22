from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import socket
import subprocess
import sys
from googlecloudsdk.api_lib.transfer import agent_pools_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from oauth2client import client as oauth2_client
def _authenticate_and_get_creds_file_path(existing_creds_file=None):
    """Ensures agent will be able to authenticate and returns creds."""
    if existing_creds_file:
        creds_file_path = _expand_path(existing_creds_file)
        if not os.path.exists(creds_file_path):
            fix_suggestion = 'Check for typos and ensure a creds file exists at the path'
            raise OSError(MISSING_CREDENTIALS_ERROR_TEXT.format(creds_file_path=creds_file_path, fix_suggestion=fix_suggestion, executed_command=_get_executed_command()))
    else:
        creds_file_path = oauth2_client._get_well_known_file()
        if not os.path.exists(creds_file_path):
            fix_suggestion = 'To generate a credentials file, please run `gcloud auth application-default login`'
            raise OSError(MISSING_CREDENTIALS_ERROR_TEXT.format(creds_file_path=creds_file_path, fix_suggestion=fix_suggestion, executed_command=_get_executed_command()))
    return creds_file_path