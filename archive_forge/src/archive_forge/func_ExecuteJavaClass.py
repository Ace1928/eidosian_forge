from __future__ import absolute_import
from __future__ import unicode_literals
import gcloud
import sys
import json
import os
import platform
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from six.moves import input
def ExecuteJavaClass(java_bin, jar_dir, main_jar, main_class, java_flags=None, main_args=None):
    """Execute a given java class within a directory of jars.

  Args:
    java_bin: str, path to the system Java binary
    jar_dir: str, directory of jars to put on class path
    main_jar: str, main jar (placed first on class path)
    main_class: str, name of the main class in the jar
    java_flags: [str], flags for the java binary
    main_args: args for the command
  """
    java_flags = java_flags or []
    main_args = main_args or []
    jar_dir_path = os.path.join(SDK_ROOT, jar_dir, '*')
    main_jar_path = os.path.join(SDK_ROOT, jar_dir, main_jar)
    classpath = main_jar_path + os.pathsep + jar_dir_path
    java_args = ['-cp', classpath] + list(java_flags) + [main_class] + list(main_args)
    _ExecuteTool(execution_utils.ArgsForExecutableTool(java_bin, *java_args))