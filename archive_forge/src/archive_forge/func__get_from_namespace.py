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
def _get_from_namespace(self, namespace, group_name):
    """Retrieves the option value from a _Namespace object.

        :param namespace: a _Namespace object
        :param group_name: a group name
        """
    names = [(group_name, self.dest)]
    current_name = (group_name, self.name)
    for opt in self.deprecated_opts:
        dname, dgroup = (opt.name, opt.group)
        if dname or dgroup:
            names.append((dgroup if dgroup else group_name, dname if dname else self.dest))
    value, loc = namespace._get_value(names, multi=self.multi, positional=self.positional, current_name=current_name)
    if self.deprecated_for_removal and (not self._logged_deprecation):
        self._logged_deprecation = True
        pretty_group = group_name or 'DEFAULT'
        if self.deprecated_reason:
            pretty_reason = ' ({})'.format(self.deprecated_reason)
        else:
            pretty_reason = ''
        format_str = 'Option "%(option)s" from group "%(group)s" is deprecated for removal%(reason)s.  Its value may be silently ignored in the future.'
        format_dict = {'option': self.dest, 'group': pretty_group, 'reason': pretty_reason}
        _report_deprecation(format_str, format_dict)
    return (value, loc)