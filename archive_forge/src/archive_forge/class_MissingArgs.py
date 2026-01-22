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
class MissingArgs(Exception):
    """Supplied arguments are not sufficient for calling a function."""

    def __init__(self, missing):
        self.missing = missing
        msg = _('Missing arguments: %s') % ', '.join(missing)
        super(MissingArgs, self).__init__(msg)