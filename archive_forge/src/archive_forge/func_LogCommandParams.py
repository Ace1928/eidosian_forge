from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
from collections import defaultdict
from functools import wraps
import logging
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import six
from six.moves import input
from six.moves import urllib
import boto
from gslib import VERSION
from gslib.metrics_tuple import Metric
from gslib.utils import system_util
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import HumanReadableToBytes
@CaptureAndLogException
def LogCommandParams(command_name=None, subcommands=None, global_opts=None, sub_opts=None, command_alias=None):
    """Logs info about the gsutil command being run.

  This only updates the collector's ga_params. The actual command metric will
  be collected once ReportMetrics() is called at shutdown.

  Args:
    command_name: str, The official command name (e.g. version instead of ver).
    subcommands: A list of subcommands as strings already validated by
      RunCommand. We do not log subcommands for the help or test commands.
    global_opts: A list of string tuples already parsed by __main__.
    sub_opts: A list of command-level options as string tuples already parsed
      by RunCommand.
    command_alias: str, The supported alias that the user inputed.
  """
    collector = MetricsCollector.GetCollector()
    if not collector:
        return
    if command_name and (not collector.GetGAParam('Command Name')):
        collector.ExtendGAParams({_GA_LABEL_MAP['Command Name']: command_name})
    if global_opts and (not collector.GetGAParam('Global Options')):
        global_opts_string = ','.join(sorted([opt[0].strip('-') for opt in global_opts]))
        collector.ExtendGAParams({_GA_LABEL_MAP['Global Options']: global_opts_string})
    command_name = collector.GetGAParam('Command Name')
    if not command_name:
        return
    if subcommands:
        full_command_name = '{0} {1}'.format(command_name, ' '.join(subcommands))
        collector.ExtendGAParams({_GA_LABEL_MAP['Command Name']: full_command_name})
    if sub_opts and (not collector.GetGAParam('Command-Level Options')):
        sub_opts_string = ','.join(sorted([opt[0].strip('-') for opt in sub_opts]))
        collector.ExtendGAParams({_GA_LABEL_MAP['Command-Level Options']: sub_opts_string})
    if command_alias and (not collector.GetGAParam('Command Alias')):
        collector.ExtendGAParams({_GA_LABEL_MAP['Command Alias']: command_alias})