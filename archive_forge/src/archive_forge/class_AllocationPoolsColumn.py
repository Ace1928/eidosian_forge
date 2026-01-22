import copy
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class AllocationPoolsColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        pool_formatted = ['%s-%s' % (pool.get('start', ''), pool.get('end', '')) for pool in self._value]
        return ','.join(pool_formatted)