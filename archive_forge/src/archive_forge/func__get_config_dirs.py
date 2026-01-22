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
def _get_config_dirs(project=None):
    """Return a list of directories where config files may be located.

    :param project: an optional project name

    If a project is specified, following directories are returned::

      ~/.${project}/
      ~/
      /etc/${project}/
      /etc/

    If a project is specified and installed from a snap package, following
    directories are also returned:

      ${SNAP_COMMON}/etc/${project}
      ${SNAP}/etc/${project}

    Otherwise, if project is not specified, these directories are returned:

      ~/
      /etc/
    """
    snap = os.environ.get('SNAP')
    snap_c = os.environ.get('SNAP_COMMON')
    cfg_dirs = [_fixpath(os.path.join('~', '.' + project)) if project else None, _fixpath('~'), os.path.join('/etc', project) if project else None, '/etc', os.path.join(snap_c, 'etc', project) if snap_c and project else None, os.path.join(snap, 'etc', project) if snap and project else None]
    return [x for x in cfg_dirs if x]