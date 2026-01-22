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
def _print_gcloud_storage_command_info(self, gcloud_command, env_variables, dry_run=False):
    logger_func = self.logger.info if dry_run else self.logger.debug
    logger_func('Gcloud Storage Command: {}'.format(' '.join(gcloud_command)))
    if env_variables:
        logger_func('Environment variables for Gcloud Storage:')
        for k, v in env_variables.items():
            logger_func('%s=%s', k, v)