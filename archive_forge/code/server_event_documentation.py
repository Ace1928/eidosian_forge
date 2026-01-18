import logging
import uuid
from cliff import columns
import iso8601
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
Show server event details.

    Specify ``--os-compute-api-version 2.21`` or higher to show event details
    for a deleted server, specified by ID only. Specify
    ``--os-compute-api-version 2.51`` or higher to show event details for
    non-admin users.
    