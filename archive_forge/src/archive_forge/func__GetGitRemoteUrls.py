import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetGitRemoteUrls(source_directory):
    """Finds the list of git remotes for the given source directory.

  Args:
    source_directory: The path to directory containing the source code.
  Returns:
    A dictionary of remote name to remote URL, empty if no remotes are found.
  """
    remote_url_config_output = _GetGitRemoteUrlConfigs(source_directory)
    if not remote_url_config_output:
        return {}
    result = {}
    config_lines = remote_url_config_output.split('\n')
    for config_line in config_lines:
        if not config_line:
            continue
        config_line_parts = config_line.split(' ')
        if len(config_line_parts) != 2:
            logging.debug('Skipping unexpected config line, incorrect segments: %s', config_line)
            continue
        remote_url_config_name = config_line_parts[0]
        remote_url = config_line_parts[1]
        remote_url_name_match = re.match(_REMOTE_URL_PATTERN, remote_url_config_name)
        if not remote_url_name_match:
            logging.debug('Skipping unexpected config line, could not match remote: %s', config_line)
            continue
        remote_url_name = remote_url_name_match.group(1)
        result[remote_url_name] = remote_url
    return result