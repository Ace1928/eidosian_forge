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
def RemoveLabelsCommon(self, cluster_ref, remove_labels):
    """Removes labels from a cluster.

    Args:
      cluster_ref: cluster to update.
      remove_labels: labels to remove.

    Returns:
      Operation ref for label set operation.
    """
    clus = None
    try:
        clus = self.GetCluster(cluster_ref)
    except apitools_exceptions.HttpNotFoundError:
        pass
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error, util.HTTP_ERROR_FORMAT)
    clus_labels = {}
    if clus.resourceLabels:
        for item in clus.resourceLabels.additionalProperties:
            clus_labels[item.key] = str(item.value)
    if not clus_labels:
        raise util.Error(NO_LABELS_ON_CLUSTER_ERROR_MSG.format(cluster=clus.name))
    for k in remove_labels:
        try:
            clus_labels.pop(k)
        except KeyError as error:
            raise util.Error(NO_SUCH_LABEL_ERROR_MSG.format(cluster=clus.name, name=k))
    labels = self.messages.SetLabelsRequest.ResourceLabelsValue()
    for k, v in sorted(six.iteritems(clus_labels)):
        labels.additionalProperties.append(labels.AdditionalProperty(key=k, value=v))
    return (labels, clus.labelFingerprint)