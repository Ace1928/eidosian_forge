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
def _all_opt_infos(self):
    """A generator function for iteration opt infos."""
    for info in self._opts.values():
        yield (info, None)
    for group in self._groups.values():
        for info in group._opts.values():
            yield (info, group)