import argparse
import io
import logging
import os
import sys
from cliff import columns as cliff_columns
from osc_lib.api import utils as api_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class HumanReadableSizeColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        """Return a formatted visibility string

        :rtype:
            A string formatted to public/private
        """
        if self._value:
            return utils.format_size(self._value)
        else:
            return ''