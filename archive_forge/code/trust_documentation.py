import datetime
import logging
from keystoneclient import exceptions as identity_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
Identity v3 Trust action implementations