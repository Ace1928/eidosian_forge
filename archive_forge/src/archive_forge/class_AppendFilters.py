import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
class AppendFilters(argparse.Action):
    filters = []

    def __call__(self, parser, namespace, values, option_string):
        AppendFilters.filters.append(values[0])