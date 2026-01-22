from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import subprocess
import sys
from googlecloudsdk.command_lib.code import run_subprocess
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
import six
def _HandleMinikubeStatusEvent(progress_bar, json_obj):
    """Handle a minikube json event."""
    if json_obj['type'] == _MINIKUBE_STEP:
        data = json_obj['data']
        if data.get('currentstep', '') != '' and data.get('totalsteps', '') != '':
            current_step = int(data['currentstep'])
            total_steps = int(data['totalsteps'])
            completion_fraction = current_step / float(total_steps)
            progress_bar.SetProgress(completion_fraction)
    elif json_obj['type'] == _MINIKUBE_DOWNLOAD_PROGRESS:
        data = json_obj['data']
        if data.get('currentstep', '') != '' and data.get('totalsteps', '') != '' and ('progress' in data):
            current_step = int(data['currentstep'])
            total_steps = int(data['totalsteps'])
            download_progress = float(data['progress'])
            completion_fraction = (current_step + download_progress) / total_steps
            progress_bar.SetProgress(completion_fraction)
    elif json_obj['type'] == _MINIKUBE_ERROR and 'exitcode' in json_obj['data']:
        data = json_obj['data']
        if 'id' in data and 'advice' in data and (data['id'] in _MINIKUBE_PASSTHROUGH_ADVICE_IDS):
            raise MinikubeStartError(data['advice'])
        else:
            exit_code = data['exitcode']
            msg = _MINIKUBE_ERROR_MESSAGES.get(exit_code, 'Unable to start Cloud Run Emulator.')
            raise MinikubeStartError(msg)