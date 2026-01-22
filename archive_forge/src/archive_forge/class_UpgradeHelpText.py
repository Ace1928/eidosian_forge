from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util.semver import SemVer
class UpgradeHelpText(object):
    """Upgrade available help text messages."""
    UPGRADE_AVAILABLE = '\n* - There is an upgrade available for your cluster(s).\n'
    SUPPORT_ENDING = '\n** - The current version of your cluster(s) will soon be out of support, please upgrade.\n'
    UNSUPPORTED = '\n*** - The current version of your cluster(s) is unsupported, please upgrade.\n'
    UPGRADE_COMMAND = '\nTo upgrade nodes to the latest available version, run\n  $ gcloud container clusters upgrade {name}'