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
def _get_gcs_json_endpoint_from_boto_config(config):
    gs_json_host = config.get('Credentials', 'gs_json_host')
    if gs_json_host:
        gs_json_port = config.get('Credentials', 'gs_json_port')
        port = ':' + gs_json_port if gs_json_port else ''
        json_api_version = config.get('Credentials', 'json_api_version', 'v1')
        return 'https://{}{}/storage/{}'.format(gs_json_host, port, json_api_version)
    return None