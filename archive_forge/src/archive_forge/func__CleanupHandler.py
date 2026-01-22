from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
import multiprocessing
import os
import signal
import socket
import stat
import sys
import textwrap
import time
import webbrowser
from six.moves import input
from six.moves.http_client import ResponseNotReady
import boto
from boto.provider import Provider
import gslib
from gslib.command import Command
from gslib.command import DEFAULT_TASK_ESTIMATION_THRESHOLD
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.cred_types import CredTypes
from gslib.exception import AbortException
from gslib.exception import CommandException
from gslib.metrics import CheckAndMaybePromptForAnalyticsEnabling
from gslib.sig_handling import RegisterSignalHandler
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils.hashing_helper import CHECK_HASH_ALWAYS
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_FAIL
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_SKIP
from gslib.utils.hashing_helper import CHECK_HASH_NEVER
from gslib.utils.parallelism_framework_util import ShouldProhibitMultiprocessing
from httplib2 import ServerNotFoundError
from oauth2client.client import HAS_CRYPTO
def _CleanupHandler(unused_signalnum, unused_handler):
    raise AbortException('User interrupted config command')