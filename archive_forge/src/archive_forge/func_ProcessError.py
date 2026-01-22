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
def ProcessError(err: BaseException, name: str='unknown', message_prefix: str='You have encountered a bug in the BigQuery CLI.') -> int:
    """Translate an error message into some printing and a return code."""
    bq_logging.ConfigurePythonLogger(FLAGS.apilog)
    logger = logging.getLogger(__name__)
    if isinstance(err, SystemExit):
        logger.exception('An error has caused the tool to exit', exc_info=err)
        return err.code
    response = []
    retcode = 1
    etype, value, tb = sys.exc_info()
    trace = ''.join(traceback.format_exception(etype, value, tb))
    contact_us_msg = _GenerateContactUsMessage()
    platform_str = GetPlatformString()
    error_details = textwrap.dedent('     ========================================\n     == Platform ==\n       %s\n     == bq version ==\n       %s\n     == Command line ==\n       %s\n     == UTC timestamp ==\n       %s\n     == Error trace ==\n     %s\n     ========================================\n     ') % (platform_str, stringutil.ensure_str(VERSION_NUMBER), [stringutil.ensure_str(item) for item in sys.argv], time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), stringutil.ensure_str(trace))
    codecs.register_error('strict', codecs.replace_errors)
    message = bq_logging.EncodeForPrinting(err)
    if isinstance(err, (bq_error.BigqueryNotFoundError, bq_error.BigqueryDuplicateError)):
        response.append('BigQuery error in %s operation: %s' % (name, message))
        retcode = 2
    elif isinstance(err, bq_error.BigqueryTermsOfServiceError):
        response.append(str(err) + '\n')
        response.append(_BIGQUERY_TOS_MESSAGE)
    elif isinstance(err, bq_error.BigqueryInvalidQueryError):
        response.append('Error in query string: %s' % (message,))
    elif isinstance(err, bq_error.BigqueryError) and (not isinstance(err, bq_error.BigqueryInterfaceError)):
        response.append('BigQuery error in %s operation: %s' % (name, message))
    elif isinstance(err, (app.UsageError, TypeError)):
        response.append(message)
    elif isinstance(err, SyntaxError) or isinstance(err, bq_error.BigquerySchemaError):
        response.append('Invalid input: %s' % (message,))
    elif isinstance(err, flags.Error):
        response.append('Error parsing command: %s' % (message,))
    elif isinstance(err, KeyboardInterrupt):
        response.append('')
    else:
        if isinstance(err, bq_error.BigqueryInterfaceError):
            message_prefix = 'Bigquery service returned an invalid reply in %s operation: %s.\n\nPlease make sure you are using the latest version of the bq tool and try again. If this problem persists, you may have encountered a bug in the bigquery client.' % (name, message)
        elif isinstance(err, oauth2client_4_0.client.Error):
            message_prefix = 'Authorization error. This may be a network connection problem, so please try again. If this problem persists, the credentials may be corrupt. Try deleting and re-creating your credentials. You can delete your credentials using "bq init --delete_credentials".\n\nIf this problem still occurs, you may have encountered a bug in the bigquery client.'
        elif isinstance(err, http.client.HTTPException) or isinstance(err, googleapiclient.errors.Error) or isinstance(err, httplib2.HttpLib2Error):
            message_prefix = 'Network connection problem encountered, please try again.\n\nIf this problem persists, you may have encountered a bug in the bigquery client.'
        message = message_prefix + ' ' + contact_us_msg
        wrap_error_message = True
        if wrap_error_message:
            message = flags.text_wrap(message)
        print(message)
        print(error_details)
        response.append('Unexpected exception in %s operation: %s' % (name, message))
    response_message = '\n'.join(response)
    wrap_error_message = True
    if wrap_error_message:
        response_message = flags.text_wrap(response_message)
    logger.exception(response_message, exc_info=err)
    print(response_message)
    return retcode