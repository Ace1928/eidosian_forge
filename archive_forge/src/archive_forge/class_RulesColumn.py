import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class RulesColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        return '\n'.join((str(v) for v in self._value))