from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.secrets import api as secrets_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.secrets import args as secrets_args
from googlecloudsdk.command_lib.secrets import log as secrets_log
from googlecloudsdk.core.console import console_io
Destroy a secret version's metadata and secret data.

  Destroy a secret version's metadata and secret data. This action is
  irreversible.

  ## EXAMPLES

  Destroy version '123' of the secret named 'my-secret':

    $ {command} 123 --secret=my-secret

  Destroy version '123' of the secret named 'my-secret' using an etag:

    $ {command} 123 --secret=my-secret --etag=\"123\"
  