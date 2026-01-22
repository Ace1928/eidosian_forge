import configparser
import os
import shlex
import subprocess
from os.path import expanduser, expandvars
from pathlib import Path
from typing import List, Optional, Union
from gitlab.const import USER_AGENT
def _parse_config(self) -> None:
    _config = configparser.ConfigParser()
    _config.read(self._files, encoding='utf-8')
    if self.gitlab_id and (not _config.has_section(self.gitlab_id)):
        raise GitlabDataError(f'A gitlab id was provided ({self.gitlab_id}) but no config section found')
    if self.gitlab_id is None:
        try:
            self.gitlab_id = _config.get('global', 'default')
        except Exception as e:
            raise GitlabIDError('Impossible to get the gitlab id (not specified in config file)') from e
    try:
        self.url = _config.get(self.gitlab_id, 'url')
    except Exception as e:
        raise GitlabDataError(f'Impossible to get gitlab details from configuration ({self.gitlab_id})') from e
    try:
        self.ssl_verify = _config.getboolean('global', 'ssl_verify')
    except ValueError:
        self.ssl_verify = _config.get('global', 'ssl_verify')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.ssl_verify = _config.getboolean(self.gitlab_id, 'ssl_verify')
    except ValueError:
        self.ssl_verify = _config.get(self.gitlab_id, 'ssl_verify')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.timeout = _config.getint('global', 'timeout')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.timeout = _config.getint(self.gitlab_id, 'timeout')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.private_token = _config.get(self.gitlab_id, 'private_token')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.oauth_token = _config.get(self.gitlab_id, 'oauth_token')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.job_token = _config.get(self.gitlab_id, 'job_token')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.http_username = _config.get(self.gitlab_id, 'http_username')
        self.http_password = _config.get(self.gitlab_id, 'http_password')
    except _CONFIG_PARSER_ERRORS:
        pass
    self._get_values_from_helper()
    try:
        self.api_version = _config.get('global', 'api_version')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.api_version = _config.get(self.gitlab_id, 'api_version')
    except _CONFIG_PARSER_ERRORS:
        pass
    if self.api_version not in ('4',):
        raise GitlabDataError(f'Unsupported API version: {self.api_version}')
    for section in ['global', self.gitlab_id]:
        try:
            self.per_page = _config.getint(section, 'per_page')
        except _CONFIG_PARSER_ERRORS:
            pass
    if self.per_page is not None and (not 0 <= self.per_page <= 100):
        raise GitlabDataError(f'Unsupported per_page number: {self.per_page}')
    try:
        self.pagination = _config.get(self.gitlab_id, 'pagination')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.order_by = _config.get(self.gitlab_id, 'order_by')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.user_agent = _config.get('global', 'user_agent')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.user_agent = _config.get(self.gitlab_id, 'user_agent')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.keep_base_url = _config.getboolean('global', 'keep_base_url')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.keep_base_url = _config.getboolean(self.gitlab_id, 'keep_base_url')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.retry_transient_errors = _config.getboolean('global', 'retry_transient_errors')
    except _CONFIG_PARSER_ERRORS:
        pass
    try:
        self.retry_transient_errors = _config.getboolean(self.gitlab_id, 'retry_transient_errors')
    except _CONFIG_PARSER_ERRORS:
        pass