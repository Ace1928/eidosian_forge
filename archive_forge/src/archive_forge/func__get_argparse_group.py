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
def _get_argparse_group(self, parser):
    if self._argparse_group is None:
        'Build an argparse._ArgumentGroup for this group.'
        self._argparse_group = parser.add_argument_group(self.title, self.help)
    return self._argparse_group