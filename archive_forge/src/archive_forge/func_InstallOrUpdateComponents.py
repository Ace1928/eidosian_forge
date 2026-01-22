from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import bootstrapping
import argparse
import os
import sys
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import platforms_install
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk import gcloud_main
def InstallOrUpdateComponents(component_ids, compile_python, update):
    """Installs or updates the given components.

  Args:
    component_ids: [str], The components to install or update.
    compile_python: bool, False if we skip compile python
    update: bool, True if we should run update, False to run install.  If there
      are no components to install, this does nothing unless in update mode (in
      which case everything gets updated).
  """
    if not update and (not component_ids):
        return
    print('\nThis will install all the core command line tools necessary for working with\nthe Google Cloud Platform.\n')
    verb = 'update' if update else 'install'
    execute_arg_list = ['--quiet', 'components', verb, '--allow-no-backup']
    if not compile_python:
        execute_arg_list.append('--no-compile-python')
    else:
        execute_arg_list.append('--compile-python')
    _CLI.Execute(execute_arg_list + component_ids)