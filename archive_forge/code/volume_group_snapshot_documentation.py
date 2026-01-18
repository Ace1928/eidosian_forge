import logging
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
Show detailed information for a volume group snapshot.

    This command requires ``--os-volume-api-version`` 3.14 or greater.
    