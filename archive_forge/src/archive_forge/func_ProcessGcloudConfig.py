import codecs
import copy
import http.client
import json
import logging
import os
import pkgutil
import platform
import sys
import textwrap
import time
import traceback
from typing import Any, Dict, List, Optional, TextIO
from absl import app
from absl import flags
from google.auth import version as google_auth_version
from google.oauth2 import credentials as google_oauth2
import googleapiclient
import httplib2
import oauth2client_4_0.client
import requests
import urllib3
from utils import bq_error
from utils import bq_logging
from pyglib import stringutil
def ProcessGcloudConfig(flag_values) -> None:
    """Processes the user's gcloud config and applies that configuration to BQ."""
    gcloud_file_name = GetGcloudConfigFilename()
    if not gcloud_file_name:
        logging.warning('Not processing gcloud config file since it is not found')
        return
    try:
        configs = _ProcessConfigSections(filename=gcloud_file_name, section_names=['billing', 'auth', 'core'])
        billing_config = configs.get('billing')
        auth_config = configs.get('auth')
        core_config = configs.get('core')
    except IOError:
        logging.warning('Could not load gcloud config data')
        return
    _UseGcloudValueIfExistsAndFlagIsDefaultValue(flag_values=flag_values, flag_name='project_id', gcloud_config_section=core_config, gcloud_property_name='project')
    _UseGcloudValueIfExistsAndFlagIsDefaultValue(flag_values=flag_values, flag_name='quota_project_id', gcloud_config_section=billing_config, gcloud_property_name='quota_project')
    _UseGcloudValueIfExistsAndFlagIsDefaultValue(flag_values=flag_values, flag_name='universe_domain', gcloud_config_section=core_config, gcloud_property_name='universe_domain')
    if not auth_config or not core_config:
        return
    try:
        access_token_file = auth_config['access_token_file']
        universe_domain = core_config['universe_domain']
    except KeyError:
        return
    if access_token_file and universe_domain:
        if not flag_values['oauth_access_token'].using_default_value or not flag_values['use_google_auth'].using_default_value:
            logging.warning('Users gcloud config file and bigqueryrc file have incompatible configurations. Defaulting to the bigqueryrc file')
            return
        logging.info('Using the gcloud configuration to get TPC authorisation from access_token_file')
        try:
            with open(access_token_file) as token_file:
                token = token_file.read().strip()
        except IOError:
            logging.warning('Could not open `access_token_file` file, ignoring gcloud settings')
        else:
            UpdateFlag(flag_values, 'oauth_access_token', token)
            UpdateFlag(flag_values, 'use_google_auth', True)