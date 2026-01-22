from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def _GetKubernetesObjectsExportConfigForClusterUpdate(options, messages):
    """Gets the KubernetesObjectsExportConfig from update options."""
    if options.kubernetes_objects_changes_target is not None or options.kubernetes_objects_snapshots_target is not None:
        changes_target = None
        snapshots_target = None
        if options.kubernetes_objects_changes_target is not None:
            changes_target = options.kubernetes_objects_changes_target
            if changes_target == 'NONE':
                changes_target = ''
        if options.kubernetes_objects_snapshots_target is not None:
            snapshots_target = options.kubernetes_objects_snapshots_target
            if snapshots_target == 'NONE':
                snapshots_target = ''
        return messages.KubernetesObjectsExportConfig(kubernetesObjectsSnapshotsTarget=snapshots_target, kubernetesObjectsChangesTarget=changes_target)