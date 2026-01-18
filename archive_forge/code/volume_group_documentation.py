import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
Failover replication for a volume group.

    This command requires ``--os-volume-api-version`` 3.38 or greater.
    