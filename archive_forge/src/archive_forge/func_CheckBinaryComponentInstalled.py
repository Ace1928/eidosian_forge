from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def CheckBinaryComponentInstalled(component_name, check_hidden=False):
    platform = platforms.Platform.Current() if config.Paths().sdk_root else None
    try:
        manager = update_manager.UpdateManager(platform_filter=platform, warn=False)
        return component_name in manager.GetCurrentVersionsInformation(include_hidden=check_hidden)
    except local_state.Error:
        log.warning('Component check failed. Could not verify SDK install path.')
        return None