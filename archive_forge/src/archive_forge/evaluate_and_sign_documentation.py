from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import platform_policy
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.binauthz import flags
from googlecloudsdk.command_lib.container.binauthz import parsing
from googlecloudsdk.command_lib.container.binauthz import sigstore_image
from googlecloudsdk.command_lib.container.binauthz import util
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.exceptions import Error
Evaluate a Binary Authorization platform policy and sign the results, if conformant.

  ## EXAMPLES

  To evaluate and sign a policy using its resource name:

    $ {command} projects/my-proj/platforms/gke/policies/my-policy
    --resource=$KUBERNETES_RESOURCE

  To evaluate the same policy using flags against multiple images:

    $ {command} my-policy --platform=gke --project=my-proj --image=$IMAGE1
    --image=$IMAGE2

  To return a modified resource with attestations added as an annotation on the
  input resource, without uploading attestations to the registry:

    $ {command} projects/my-proj/platforms/gke/policies/my-policy
    --resource=$KUBERNETES_RESOURCE --output-file=$MODIFIED_RESOURCE --no-upload

  To upload attestations using Docker credentials located in a custom directory:

    $ {command} projects/my-proj/platforms/gke/policies/my-policy
    --image=$IMAGE --use-docker-creds --docker-config-dir=$CUSTOM_DIR
  