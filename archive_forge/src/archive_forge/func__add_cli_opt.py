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
def _add_cli_opt(self, opt, group):
    if {'opt': opt, 'group': group} in self._cli_opts:
        return
    if opt.positional:
        self._cli_opts.append({'opt': opt, 'group': group})
    else:
        self._cli_opts.appendleft({'opt': opt, 'group': group})