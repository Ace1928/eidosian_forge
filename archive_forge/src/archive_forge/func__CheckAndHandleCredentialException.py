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
def _CheckAndHandleCredentialException(e, args):
    if not boto_util.HasConfiguredCredentials() and (not boto.config.get_value('Tests', 'bypass_anonymous_access_warning', False)):
        if system_util.InvokedViaCloudSdk():
            message = '\n'.join(textwrap.wrap('You are attempting to access protected data with no configured credentials. Please visit https://cloud.google.com/console#/project and sign up for an account, and then run the "gcloud auth login" command to configure gsutil to use these credentials.'))
        else:
            message = '\n'.join(textwrap.wrap('You are attempting to access protected data with no configured credentials. Please visit https://cloud.google.com/console#/project and sign up for an account, and then run the "gsutil config" command to configure gsutil to use these credentials.'))
        _OutputAndExit(message=message, exception=e)
    elif e.reason and (e.reason == 'AccountProblem' or e.reason == 'Account disabled.' or 'account for the specified project has been disabled' in e.reason) and (','.join(args).find('gs://') != -1):
        _OutputAndExit('\n'.join(textwrap.wrap(_ConstructAccountProblemHelp(e.reason))), exception=e)