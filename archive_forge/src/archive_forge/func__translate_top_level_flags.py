from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
import re
import subprocess
from boto import config
from gslib import exception
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
def _translate_top_level_flags(self):
    """Translates gsutil's top level flags.

    Gsutil specifies the headers (-h) and boto config (-o) as top level flags
    as well, but we handle those separately.

    Returns:
      A tuple. The first item is a list of top level flags that can be appended
        to the gcloud storage command. The second item is a dict of environment
        variables that can be set for the gcloud storage command execution.
    """
    top_level_flags = []
    env_variables = {'CLOUDSDK_METRICS_ENVIRONMENT': 'gsutil_shim', 'CLOUDSDK_STORAGE_RUN_BY_GSUTIL_SHIM': 'True'}
    if self.debug >= 3:
        top_level_flags.extend(['--verbosity', 'debug'])
    if self.debug == 4:
        top_level_flags.append('--log-http')
    if self.quiet_mode:
        top_level_flags.append('--no-user-output-enabled')
    if self.user_project:
        top_level_flags.append('--billing-project=' + self.user_project)
    if self.trace_token:
        top_level_flags.append('--trace-token=' + self.trace_token)
    if constants.IMPERSONATE_SERVICE_ACCOUNT:
        top_level_flags.append('--impersonate-service-account=' + constants.IMPERSONATE_SERVICE_ACCOUNT)
    should_use_rsync_override = self.command_name == 'rsync' and (not (config.get('GSUtil', 'parallel_process_count') == '1' and config.get('GSUtil', 'thread_process_count') == '1'))
    if not (self.parallel_operations or should_use_rsync_override):
        env_variables['CLOUDSDK_STORAGE_THREAD_COUNT'] = '1'
        env_variables['CLOUDSDK_STORAGE_PROCESS_COUNT'] = '1'
    return (top_level_flags, env_variables)