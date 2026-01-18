from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
Set usage reporting bucket for a project.

    *{command}* configures usage reporting for projects.

  Setting usage reporting will cause a log of usage per resource to be
  written to a specified Google Cloud Storage bucket daily.

  For example, to write daily logs of the form usage_gce_YYYYMMDD.csv
  to the bucket `my-bucket`, run:

    $ gcloud compute project-info set-usage-bucket --bucket=gs://my-bucket

  To disable this feature, issue the command:

    $ gcloud compute project-info set-usage-bucket --no-bucket
  