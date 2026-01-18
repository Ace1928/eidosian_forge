from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
List Google Domains registrations importable into Cloud Domains.

  List Google Domains registrations that can be imported to a Cloud Domains
  project.

  Registrations with an IMPORTABLE resource state can be imported from
  Google Domains registrar to Cloud Domains.

  Registrations with a SUSPENDED, EXPIRED, or DELETED resource state must have
  their states resolved with Google Domains registrar to be imported.

  Registrations with an UNSUPPORTED resource state are not currently supported
  for import.

  ## EXAMPLES

  To list Google Domains registrations that can be imported, run:

    $ {command}
  