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
def _unset_defaults_and_overrides(self):
    """Unset any default or override on all options."""
    for info, group in self._all_opt_infos():
        info.pop('default', None)
        info.pop('override', None)