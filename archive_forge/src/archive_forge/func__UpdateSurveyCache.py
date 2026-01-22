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
def _UpdateSurveyCache():
    """Records the time user answers survey."""
    with survey_check.PromptRecord() as pr:
        pr.last_answer_survey_time = time.time()