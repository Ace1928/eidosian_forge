import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class EncryptionInfoColumn(cliff_columns.FormattableColumn):
    """Formattable column for encryption info column.

    Unlike the parent FormattableColumn class, the initializer of the
    class takes encryption_data as the second argument.
    osc_lib.utils.get_item_properties instantiate cliff FormattableColumn
    object with a single parameter "column value", so you need to pass
    a partially initialized class like
    ``functools.partial(EncryptionInfoColumn encryption_data)``.
    """

    def __init__(self, value, encryption_data=None):
        super(EncryptionInfoColumn, self).__init__(value)
        self._encryption_data = encryption_data or {}

    def _get_encryption_info(self):
        type_id = self._value
        return self._encryption_data.get(type_id)

    def human_readable(self):
        encryption_info = self._get_encryption_info()
        if encryption_info:
            return utils.format_dict(encryption_info)
        else:
            return '-'

    def machine_readable(self):
        return self._get_encryption_info()