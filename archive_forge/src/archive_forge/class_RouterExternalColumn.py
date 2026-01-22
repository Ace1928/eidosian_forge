from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class RouterExternalColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        return 'External' if self._value else 'Internal'