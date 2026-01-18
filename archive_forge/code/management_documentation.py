from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import log
Configure management settings of a Cloud Domains registration.

  Configure management settings of a registration. This includes settings
  related to transfers, billing and renewals of a registration.

  ## EXAMPLES

  To start an interactive flow to configure management settings for
  ``example.com'', run:

    $ {command} example.com

  To unlock a transfer lock of a registration for ``example.com'', run:

    $ {command} example.com --transfer-lock-state=unlocked

  To disable automatic renewals for ``example.com'', run:

    $ {command} example.com --preferred-renewal-method=renewal-disabled
  