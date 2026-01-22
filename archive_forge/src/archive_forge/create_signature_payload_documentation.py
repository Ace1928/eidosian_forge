from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.binauthz import flags as binauthz_flags
from googlecloudsdk.command_lib.container.binauthz import util as binauthz_command_util
Create a JSON container image signature object.

  Given a container image URL specified by the manifest digest, this command
  will produce a JSON object whose signature is expected by Cloud Binary
  Authorization.

  ## EXAMPLES

  To output serialized JSON to sign, run:

      $ {command} \
          --artifact-url="gcr.io/example-project/example-image@sha256:abcd"
  