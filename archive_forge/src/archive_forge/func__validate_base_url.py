import collections.abc
import configparser
import enum
import getpass
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from functools import reduce
from typing import (
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue, Int32Value, StringValue
import wandb
import wandb.env
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import UsageError
from wandb.proto import wandb_settings_pb2
from wandb.sdk.internal.system.env_probe_helpers import is_aws_lambda
from wandb.sdk.lib import filesystem
from wandb.sdk.lib._settings_toposort_generated import SETTINGS_TOPOLOGICALLY_SORTED
from wandb.sdk.wandb_setup import _EarlyLogger
from .lib import apikey
from .lib.gitlib import GitRepo
from .lib.ipython import _get_python_type
from .lib.runid import generate_id
@staticmethod
def _validate_base_url(value: Optional[str]) -> bool:
    """Validate the base url of the wandb server.

        param value: URL to validate

        Based on the Django URLValidator, but with a few additional checks.

        Copyright (c) Django Software Foundation and individual contributors.
        All rights reserved.

        Redistribution and use in source and binary forms, with or without modification,
        are permitted provided that the following conditions are met:

            1. Redistributions of source code must retain the above copyright notice,
               this list of conditions and the following disclaimer.

            2. Redistributions in binary form must reproduce the above copyright
               notice, this list of conditions and the following disclaimer in the
               documentation and/or other materials provided with the distribution.

            3. Neither the name of Django nor the names of its contributors may be used
               to endorse or promote products derived from this software without
               specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
        ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
        ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
        (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
        LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
        ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """
    if value is None:
        return True
    ul = '¡-\uffff'
    ipv4_re = '(?:0|25[0-5]|2[0-4][0-9]|1[0-9]?[0-9]?|[1-9][0-9]?)(?:\\.(?:0|25[0-5]|2[0-4][0-9]|1[0-9]?[0-9]?|[1-9][0-9]?)){3}'
    ipv6_re = '\\[[0-9a-f:.]+\\]'
    hostname_re = '[a-z' + ul + '0-9](?:[a-z' + ul + '0-9-]{0,61}[a-z' + ul + '0-9])?'
    domain_re = '(?:\\.(?!-)[a-z' + ul + '0-9-]{1,63}(?<!-))*'
    tld_re = '\\.(?!-)(?:[a-z' + ul + '-]{2,63}|xn--[a-z0-9]{1,59})(?<!-)\\.?'
    host_re = '(' + hostname_re + domain_re + f'({tld_re})?' + '|localhost)'
    regex = re.compile('^(?:[a-z0-9.+-]*)://(?:[^\\s:@/]+(?::[^\\s:@/]*)?@)?(?:' + ipv4_re + '|' + ipv6_re + '|' + host_re + ')(?::[0-9]{1,5})?(?:[/?#][^\\s]*)?\\Z', re.IGNORECASE)
    schemes = {'http', 'https'}
    unsafe_chars = frozenset('\t\r\n')
    scheme = value.split('://')[0].lower()
    split_url = urlsplit(value)
    parsed_url = urlparse(value)
    if re.match('.*wandb\\.ai[^\\.]*$', value) and 'api.' not in value:
        raise UsageError(f'{value} is not a valid server address, did you mean https://api.wandb.ai?')
    elif re.match('.*wandb\\.ai[^\\.]*$', value) and scheme != 'https':
        raise UsageError('http is not secure, please use https://api.wandb.ai')
    elif parsed_url.netloc == '':
        raise UsageError(f'Invalid URL: {value}')
    elif unsafe_chars.intersection(value):
        raise UsageError('URL cannot contain unsafe characters')
    elif scheme not in schemes:
        raise UsageError('URL must start with `http(s)://`')
    elif not regex.search(value):
        raise UsageError(f'{value} is not a valid server address')
    elif split_url.hostname is None or len(split_url.hostname) > 253:
        raise UsageError('hostname is invalid')
    return True