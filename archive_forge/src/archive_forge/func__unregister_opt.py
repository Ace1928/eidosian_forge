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
def _unregister_opt(self, opt):
    """Remove an opt from this group.

        :param opt: an Opt object
        """
    if opt.dest in self._opts:
        del self._opts[opt.dest]