from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _GetDefaultCliCommandVersion():
    """Return the default CLI command version."""
    if _IsRunningUnderTest():
        return TEST_CLI_VERSION_TEST
    version = config.CLOUD_SDK_VERSION
    if version != TEST_CLI_VERSION_HEAD:
        return version
    try:
        from googlecloudsdk.core.updater import update_manager
        manager = update_manager.UpdateManager()
        components = manager.GetCurrentVersionsInformation()
        version = components['core']
    except (KeyError, exceptions.Error):
        pass
    return version