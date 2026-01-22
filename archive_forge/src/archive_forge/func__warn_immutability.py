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
def _warn_immutability(self):
    """Check immutable opts have not changed.

        _do_get won't return the new values but presumably someone changed the
        config file expecting them to change so we should warn them they won't.
        """
    for info, group in self._all_opt_infos():
        opt = info['opt']
        if opt.mutable:
            continue
        groupname = group.name if group else 'DEFAULT'
        try:
            old, _ = opt._get_from_namespace(self._namespace, groupname)
        except KeyError:
            old = None
        try:
            new, _ = opt._get_from_namespace(self._mutable_ns, groupname)
        except KeyError:
            new = None
        if old != new:
            LOG.warning('Ignoring change to immutable option %(group)s.%(option)s', {'group': groupname, 'option': opt.name})