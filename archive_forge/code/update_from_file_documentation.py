from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkeonprem import operations
from googlecloudsdk.api_lib.container.gkeonprem import vmware_clusters as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.container.gkeonprem import constants
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
Update an Anthos on VMware user cluster using a configuration file.