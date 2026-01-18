from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.iot import resource_args
Set the IAM policy for a device registry.

  This command replaces the existing IAM policy for a device registry, given
  a REGISTRY and a file encoded in JSON or YAML that contains the IAM
  policy. If the given policy file specifies an "etag" value, then the
  replacement will succeed only if the policy already in place matches that
  etag. (An etag obtained via $ gcloud iot registries get-iam-policy will
  prevent the replacement if the policy for the device registry has been
  subsequently updated.) A policy file that does not contain an etag value will
  replace any existing policy for the device registry.
  