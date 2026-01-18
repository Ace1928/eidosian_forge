from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import resource_args
Set the primary version of a key.

  Sets the specified version as the primary version of the given key.
  The version is specified by its version number assigned on creation.

  ## EXAMPLES

  The following command sets version 9 as the primary version of the
  key `samwise` within keyring `fellowship` and location `global`:

    $ {command} samwise --version=9 --keyring=fellowship --location=global
  