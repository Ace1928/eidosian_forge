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
def _get_argparse_prefix(self, prefix, group_name):
    """Build a prefix for the CLI option name, if required.

        CLI options in a group are prefixed with the group's name in order
        to avoid conflicts between similarly named options in different
        groups.

        :param prefix: an existing prefix to append to (for example 'no' or '')
        :param group_name: an optional group name
        :returns: a CLI option prefix including the group name, if appropriate
        """
    if group_name is not None:
        return group_name + '-' + prefix
    else:
        return prefix