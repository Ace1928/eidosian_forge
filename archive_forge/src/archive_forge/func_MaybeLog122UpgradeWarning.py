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
def MaybeLog122UpgradeWarning(cluster):
    """Logs deprecation warning for GKE v1.22 upgrades."""
    if cluster is not None:
        cmv = SemVer(cluster.currentMasterVersion)
        if cmv >= SemVer('1.22.0-gke.0'):
            return
    log.status.Print('Upcoming breaking change: Starting with v1.22, Kubernetes has removed several v1beta1 APIs for more stable v1 APIs. Read more about this change - https://cloud.google.com/kubernetes-engine/docs/deprecations/apis-1-22. Please ensure that your cluster is not using any deprecated v1beta1 APIs prior to upgrading to GKE 1.22.')