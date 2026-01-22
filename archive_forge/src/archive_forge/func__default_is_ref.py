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
def _default_is_ref(self):
    """Check if default is a reference to another var."""
    if isinstance(self.default, str):
        tmpl = self.default.replace('\\$', '').replace('$$', '')
        return '$' in tmpl
    return False