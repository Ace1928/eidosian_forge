from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _validate_versions(self, membership, cluster_v, new_v):
    if cluster_v == new_v:
        log.status.Print('Membership {} already has version {} of the {} Feature installed.'.format(membership, cluster_v, self.feature.display_name))
        return False
    return True