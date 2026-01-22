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
def _get_opt_info(self, opt_name, group=None):
    """Return the (opt, override, default) dict for an opt.

        :param opt_name: an opt name/dest
        :param group: an optional group name or OptGroup object
        :raises: NoSuchOptError, NoSuchGroupError
        """
    if group is None:
        opts = self._opts
    else:
        group = self._get_group(group)
        opts = group._opts
    if opt_name not in opts:
        real_opt_name, real_group_name = self._find_deprecated_opts(opt_name, group=group)
        if not real_opt_name:
            raise NoSuchOptError(opt_name, group)
        log_real_group_name = real_group_name or 'DEFAULT'
        dep_message = 'Config option %(dep_group)s.%(dep_option)s  is deprecated. Use option %(group)s.%(option)s instead.'
        LOG.warning(dep_message, {'dep_option': opt_name, 'dep_group': group, 'option': real_opt_name, 'group': log_real_group_name})
        opt_name = real_opt_name
        if real_group_name:
            group = self._get_group(real_group_name)
            opts = group._opts
    return opts[opt_name]