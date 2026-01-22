import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
def _report_deprecation(format_str, format_dict):
    """Report use of a deprecated option

    Uses versionutils from oslo.log if it is available.  If not, logs
    a simple warning message.

    :param format_str: The message to use for the report
    :param format_dict: A dict containing keys for any parameters in format_str
    """
    if oslo_log:
        from oslo_log import versionutils
        versionutils.report_deprecated_feature(LOG, format_str, format_dict)
    else:
        LOG.warning(format_str, format_dict)