from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iot import resource_args
Get the IAM policy for a device registry.

  This command gets the IAM policy for a device registry. If formatted as
  JSON, the output can be edited and used as a policy file for
  set-iam-policy. The output includes an "etag" field identifying the version
  emitted and allowing detection of concurrent policy updates; see
  $ gcloud iot registries set-iam-policy for additional details.
  