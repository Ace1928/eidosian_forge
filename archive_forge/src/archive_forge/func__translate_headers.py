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
def _translate_headers(self, headers=None, unset=False):
    """Translates gsutil headers to equivalent gcloud storage flags.

    Args:
      headers (dict|None): If absent, extracts headers from command instance.
      unset (bool): Yield metadata clear flags instead of setter flags.

    Returns:
      List[str]: Translated flags for gcloud.

    Raises:
      GcloudStorageTranslationError: Could not translate flag.
    """
    flags = []
    headers_to_translate = headers if headers is not None else self.headers
    additional_headers = []
    for raw_header_key, header_value in headers_to_translate.items():
        lowercase_header_key = raw_header_key.lower()
        if lowercase_header_key == 'x-goog-api-version':
            continue
        flag = get_flag_from_header(raw_header_key, header_value, unset=unset)
        if self.command_name in COMMANDS_SUPPORTING_ALL_HEADERS:
            if flag:
                flags.append(flag)
        elif self.command_name in PRECONDITONS_ONLY_SUPPORTED_COMMANDS and lowercase_header_key in PRECONDITIONS_HEADERS:
            flags.append(flag)
        if not flag:
            self.logger.warn('Header {}:{} cannot be translated to a gcloud storage equivalent flag. It is being treated as an arbitrary request header.'.format(raw_header_key, header_value))
            additional_headers.append('{}={}'.format(raw_header_key, header_value))
    if additional_headers:
        flags.append('--additional-headers=' + ','.join(additional_headers))
    return flags