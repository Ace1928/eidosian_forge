from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import errno
import getopt
import logging
import os
import re
import signal
import socket
import sys
import textwrap
import traceback
import six
from six.moves import configparser
from six.moves import range
from google.auth import exceptions as google_auth_exceptions
import gslib.exception
from gslib.exception import CommandException
from gslib.exception import ControlCException
from gslib.utils.version_check import check_python_version_support
from gslib.utils.arg_helper import GetArgumentsAndOptions
from gslib.utils.user_agent_helper import GetUserAgent
import boto
import gslib
from gslib.utils import system_util, text_util
from gslib import metrics
import httplib2
import oauth2client
from google_reauth import reauth_creds
from google_reauth import errors as reauth_errors
from gslib import context_config
from gslib import wildcard_iterator
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import ProjectIdException
from gslib.cloud_api import ServiceException
from gslib.command_runner import CommandRunner
import apitools.base.py.exceptions as apitools_exceptions
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import InitializeSignalHandling
from gslib.sig_handling import RegisterSignalHandler
def _OutputAndExit(message, exception=None):
    """Outputs message to stderr and exits gsutil with code 1.

  This function should only be called in single-process, single-threaded mode.

  Args:
    message: Message to print to stderr.
    exception: The exception that caused gsutil to fail.
  """
    if debug_level >= constants.DEBUGLEVEL_DUMP_REQUESTS or test_exception_traces:
        stack_trace = traceback.format_exc()
        err = 'DEBUG: Exception stack trace:\n    %s\n%s\n' % (re.sub('\\n', '\n    ', stack_trace), message)
    else:
        err = '%s\n' % message
    try:
        text_util.print_to_fd(err, end='', file=sys.stderr)
    except UnicodeDecodeError:
        sys.stderr.write(err)
    if exception:
        metrics.LogFatalError(exception)
    sys.exit(1)