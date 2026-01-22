from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
Show detailed information for a block storage cluster.

    This command requires ``--os-volume-api-version`` 3.7 or greater.
    