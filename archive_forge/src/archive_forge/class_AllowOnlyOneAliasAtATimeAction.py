import argparse
import csv
import glob
from importlib import util as importlib_util
import itertools
import logging
import os
import pkgutil
import sys
from oslo_utils import importutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions as exc
import manilaclient.extension
from manilaclient.v2 import shell as shell_v2
class AllowOnlyOneAliasAtATimeAction(argparse.Action):
    """Allows only one alias of argument to be used at a time."""

    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(self, 'calls'):
            self.calls = {}
        if self.dest not in self.calls:
            self.calls[self.dest] = set()
        local_values = sorted(values) if isinstance(values, list) else values
        self.calls[self.dest].add(str(local_values))
        if len(self.calls[self.dest]) == 1:
            setattr(namespace, self.dest, local_values)
        else:
            msg = 'Only one alias is allowed at a time.'
            raise argparse.ArgumentError(self, msg)