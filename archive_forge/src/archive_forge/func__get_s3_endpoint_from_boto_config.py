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
def _get_s3_endpoint_from_boto_config(config):
    s3_host = config.get('Credentials', 's3_host')
    if s3_host:
        s3_port = config.get('Credentials', 's3_port')
        port = ':' + s3_port if s3_port else ''
        return 'https://{}{}'.format(s3_host, port)
    return None