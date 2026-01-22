from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import config as configuration
from googlecloudsdk.core import config as gcloud_config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.util import encoding
import six
def _AppendMetricsEnvironment(tag):
    """Appends tag to the Cloud SDK metrics environment tag.

  The metrics/environment tag is sent via the useragent. This tag is visible in
  metrics for all gcloud commands executed by the calling command.

  Args:
    tag: The string to append to the metrics/environment tag.
  """
    metrics_environment = properties.VALUES.metrics.environment.Get() or ''
    if metrics_environment:
        metrics_environment += '.'
    metrics_environment += tag
    encoding.SetEncodedValue(os.environ, 'CLOUDSDK_METRICS_ENVIRONMENT', metrics_environment)