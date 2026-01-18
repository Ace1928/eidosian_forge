import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def update_and_persist_config(profile, databricks_config):
    """
    Takes a DatabricksConfig and adds the in memory contents to the persisted version of the
    config. This will overwrite any other config that was persisted to the file system under the
    same profile.

    Args:
        databricks_config: DatabricksConfig
    """
    profile = profile if profile else DEFAULT_SECTION
    raw_config = _fetch_from_fs()
    _create_section_if_absent(raw_config, profile)
    _set_option(raw_config, profile, HOST, databricks_config.host)
    _set_option(raw_config, profile, USERNAME, databricks_config.username)
    _set_option(raw_config, profile, PASSWORD, databricks_config.password)
    _set_option(raw_config, profile, TOKEN, databricks_config.token)
    _set_option(raw_config, profile, REFRESH_TOKEN, databricks_config.refresh_token)
    _set_option(raw_config, profile, INSECURE, databricks_config.insecure)
    _set_option(raw_config, profile, JOBS_API_VERSION, databricks_config.jobs_api_version)
    _overwrite_config(raw_config)