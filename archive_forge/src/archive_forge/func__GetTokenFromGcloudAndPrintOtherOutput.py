import logging
import os
import subprocess
from typing import Iterator, List, Optional
from google.oauth2 import credentials as google_oauth2
import bq_auth_flags
import bq_flags
import bq_utils
from utils import bq_error
def _GetTokenFromGcloudAndPrintOtherOutput(cmd: List[str]) -> Optional[str]:
    """Returns a token or prints other messages from the given gcloud command."""
    try:
        token = None
        for output in _RunGcloudCommand(cmd):
            if output and ' ' not in output:
                token = output
                break
            else:
                print(output)
        return token
    except bq_error.BigqueryError as e:
        single_line_error_msg = str(e).replace('\n', '')
        if 'security key' in single_line_error_msg:
            raise bq_error.BigqueryError('Access token has expired. Did you touch the security key within the timeout window?\n' + _GetReauthMessage())
        elif 'Refresh token has expired' in single_line_error_msg:
            raise bq_error.BigqueryError('Refresh token has expired. ' + _GetReauthMessage())
        elif 'do not support refresh tokens' in single_line_error_msg:
            return None
        else:
            raise bq_error.BigqueryError('Error retrieving auth credentials from gcloud: %s' % str(e))
    except Exception as e:
        single_line_error_msg = str(e).replace('\n', '')
        if "No such file or directory: 'gcloud'" in single_line_error_msg:
            raise bq_error.BigqueryError("'gcloud' not found but is required for authentication. To install, follow these instructions: https://cloud.google.com/sdk/docs/install")
        raise bq_error.BigqueryError('Error retrieving auth credentials from gcloud: %s' % str(e))