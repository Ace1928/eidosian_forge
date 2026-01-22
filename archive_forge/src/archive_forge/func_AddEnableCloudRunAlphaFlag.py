from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddEnableCloudRunAlphaFlag(parser):
    """Adds the --enable-cloud-run-alpha flags to parser."""
    help_text = 'Enable Cloud Run alpha features on this cluster. Selecting this\noption will result in the cluster having all Cloud Run alpha API groups and\nfeatures turned on.\n\nCloud Run alpha clusters are not covered by the Cloud Run SLA and should not be\nused for production workloads.'
    _AddLegacyCloudRunFlag(parser, '--enable-{0}-alpha', action='store_true', help=help_text)