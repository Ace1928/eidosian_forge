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
def AddKubernetesObjectsExportConfig(parser, for_create=False):
    """Adds kubernetes-objects-changes-target and kubernetes-objects-snapshots-target flags to parser."""
    help_text = 'Set kubernetes objects changes target [Currently only CLOUD_LOGGING value is supported].\n  '
    validation_description = 'Only value CLOUD_LOGGING is accepted'
    regexp = '^CLOUD_LOGGING$|^NONE$'
    if for_create:
        regexp = '^CLOUD_LOGGING$'
    type_ = arg_parsers.RegexpValidator(regexp, validation_description)
    group = parser.add_group(hidden=True)
    group.add_argument('--kubernetes-objects-changes-target', default=None, type=type_, help=help_text)
    help_text = 'Set kubernetes objects snapshots target [Currently only CLOUD_LOGGING value is supported].\n  '
    group.add_argument('--kubernetes-objects-snapshots-target', default=None, type=type_, help=help_text)