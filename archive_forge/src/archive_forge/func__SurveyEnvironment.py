from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import platform
import socket
import time
from googlecloudsdk.command_lib.survey import question
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.survey import survey_check
from googlecloudsdk.core.util import platforms
from six.moves import http_client as httplib
def _SurveyEnvironment():
    """Gets user's environment."""
    install_type = 'Google' if socket.gethostname().endswith('.google.com') else 'External'
    env = {'install_type': install_type, 'cid': config.GetCID(), 'user_agent': metrics.GetUserAgent(), 'release_channel': config.INSTALLATION_CONFIG.release_channel, 'python_version': platform.python_version(), 'environment': properties.GetMetricsEnvironment(), 'environment_version': properties.VALUES.metrics.environment_version.Get()}
    return [{'key': k, 'value': v} for k, v in env.items() if v is not None]