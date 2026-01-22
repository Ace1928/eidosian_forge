from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
@base.Deprecate(is_removed=True, warning='This command is deprecated. See https://cloud.google.com/domains/docs/deprecations/feature-deprecations.', error='This command has been removed. See https://cloud.google.com/domains/docs/deprecations/feature-deprecations.')
class ListImportableDomains(base.ListCommand):
    """List Google Domains registrations importable into Cloud Domains.

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
  """

    @staticmethod
    def Args(parser):
        resource_args.AddLocationResourceArg(parser, 'to import to')
        parser.display_info.AddTransforms({'price': util.TransformMoneyType})
        parser.display_info.AddFormat(_FORMAT)
        base.URI_FLAG.RemoveFromParser(parser)

    def Run(self, args):
        api_version = registrations.GetApiVersionFromArgs(args)
        client = registrations.RegistrationsClient(api_version)
        location_ref = args.CONCEPTS.location.Parse()
        return client.RetrieveImportableDomains(location_ref, limit=args.limit, page_size=args.page_size, batch_size=util.GetListBatchSize(args))