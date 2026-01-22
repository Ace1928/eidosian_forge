from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def _is_job_done(self, job_json, job_state, job_error, timed_out):
    """ return (done, message, error)
            done is True to indicate that the job is complete, or failed, or timed out
            done is False when the job is still running
        """
    done, error = (False, None)
    message = job_json.get('message', '') if job_json else None
    if job_state == 'failure':
        error = message
        message = None
        done = True
    elif job_state not in ('queued', 'running', None):
        error = job_error
        done = True
    elif timed_out:
        self.log_error(0, 'Timeout error: Process still running')
        error = 'Timeout error: Process still running'
        if job_error is not None:
            error += ' - %s' % job_error
        done = True
    return (done, message, error)