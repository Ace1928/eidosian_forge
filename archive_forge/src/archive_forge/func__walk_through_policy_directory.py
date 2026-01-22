import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
@staticmethod
def _walk_through_policy_directory(path, func, *args):
    if not os.path.isdir(path):
        raise ValueError('%s is not a directory' % path)
    policy_files = next(os.walk(path))[2]
    policy_files.sort()
    for policy_file in [p for p in policy_files if not p.startswith('.')]:
        func(os.path.join(path, policy_file), *args)