from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def ShowV1DeprecationWarning(plural=False):
    message = 'Upgrade your First Generation instance{} to Second Generation before we auto-upgrade {} on March 4, 2020, ahead of the full decommission of First Generation on March 25, 2020.'
    if plural:
        log.warning(message.format('s', 'them'))
    else:
        log.warning(message.format('', 'it'))