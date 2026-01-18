import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
def validate_name_args(positional_name, optional_name):
    if optional_name:
        print(NAME_DEPRECATION_WARNING)
    if positional_name and optional_name:
        raise DuplicateArgs('<name>', (positional_name, optional_name))