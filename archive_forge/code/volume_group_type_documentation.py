import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
Show detailed information for a volume group type.

    This command requires ``--os-volume-api-version`` 3.11 or greater.
    