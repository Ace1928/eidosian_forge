from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
class ResetAuthorizationCode(base.DescribeCommand):
    """Resets authorization code of a specific Cloud Domains registration.

  Resets authorization code of a specific registration.

  You can call this API only after 60 days have elapsed since initial
  registration.

  ## EXAMPLES

  To reset authorization code of ``example.com'', run:

    $ {command} example.com
  """

    @staticmethod
    def Args(parser):
        resource_args.AddRegistrationResourceArg(parser, 'to reset authorization code for')

    def Run(self, args):
        """Run reset authorization code command."""
        api_version = registrations.GetApiVersionFromArgs(args)
        client = registrations.RegistrationsClient(api_version)
        args.registration = util.NormalizeResourceName(args.registration)
        registration_ref = args.CONCEPTS.registration.Parse()
        registration = client.Get(registration_ref)
        util.AssertRegistrationOperational(api_version, registration)
        return client.ResetAuthorizationCode(registration_ref)