from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Add or update project-wide metadata.

    *{command}* can be used to add or update project-wide
  metadata. Every instance has access to a metadata server that
  can be used to query metadata that has been set through this
  tool. Project-wide metadata entries are visible to all
  instances. To set metadata for individual instances, use
  `gcloud compute instances add-metadata`. For information on
  metadata, see
  [](https://cloud.google.com/compute/docs/metadata)

  Only metadata keys that are provided are mutated. Existing
  metadata entries will remain unaffected.

  If you are using this command to manage SSH keys for your project, please note
  the
  [risks](https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys#risks)
  of manual SSH key management as well as the required format for SSH key
  metadata, available at
  [](https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys)
  