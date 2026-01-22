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
def _diff_ns(self, old_ns, new_ns):
    """Compare mutable option values between two namespaces.

        This can be used to only reconfigure stateful sessions when necessary.

        :return {(None or 'group', 'optname'): (old_value, new_value), ... }
        """
    diff = {}
    for info, group in self._all_opt_infos():
        opt = info['opt']
        if not opt.mutable:
            continue
        groupname = group.name if group else None
        try:
            old, _ = opt._get_from_namespace(old_ns, groupname)
        except KeyError:
            old = None
        try:
            new, _ = opt._get_from_namespace(new_ns, groupname)
        except KeyError:
            new = None
        if old != new:
            diff[groupname, opt.name] = (old, new)
    return diff