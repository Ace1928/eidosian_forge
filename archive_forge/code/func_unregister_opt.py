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
@__clear_cache
def unregister_opt(self, opt, group=None):
    """Unregister an option.

        :param opt: an Opt object
        :param group: an optional OptGroup object or group name
        :raises: ArgsAlreadyParsedError, NoSuchGroupError
        """
    if self._args is not None:
        raise ArgsAlreadyParsedError('reset before unregistering options')
    remitem = None
    for item in self._cli_opts:
        if item['opt'].dest == opt.dest and (group is None or self._get_group(group).name == item['group'].name):
            remitem = item
            break
    if remitem is not None:
        self._cli_opts.remove(remitem)
    if group is not None:
        self._get_group(group)._unregister_opt(opt)
    elif opt.dest in self._opts:
        del self._opts[opt.dest]