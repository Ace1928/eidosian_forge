from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc.batches import batch_message_factory
class BatchesCreateRequestFactory(object):
    """Factory class handling BatchesCreateRequest message.

  Factory class for configure argument parser and create
  BatchesCreateRequest message from parsed argument.
  """

    def __init__(self, dataproc, batch_message_factory_override=None):
        """Factory for BatchesCreateRequest message.

    Only handles general submit flags added by this class. User needs to
    provide job specific message when creating the request message.

    Args:
      dataproc: A api_lib.dataproc.Dataproc instance.
      batch_message_factory_override: Override BatchMessageFactory instance.
    """
        self.dataproc = dataproc
        self.batch_message_factory = batch_message_factory_override
        if not self.batch_message_factory:
            self.batch_message_factory = batch_message_factory.BatchMessageFactory(self.dataproc)

    def GetRequest(self, args, batch_job):
        """Creates a BatchesCreateRequest message.

    Creates a BatchesCreateRequest message. The factory only handles the
    arguments added in AddArguments function. User needs to provide job
    specific message instance.

    Args:
      args: Parsed arguments.
      batch_job: A batch job typed message instance.

    Returns:
      BatchesCreateRequest: A configured BatchesCreateRequest.
    """
        kwargs = {}
        kwargs['parent'] = args.CONCEPTS.region.Parse().RelativeName()
        kwargs['requestId'] = args.request_id
        if not kwargs['requestId']:
            kwargs['requestId'] = util.GetUniqueId()
        kwargs['batchId'] = args.batch
        if not kwargs['batchId']:
            kwargs['batchId'] = kwargs['requestId']
        kwargs['batch'] = self.batch_message_factory.GetMessage(args, batch_job)
        return self.dataproc.messages.DataprocProjectsLocationsBatchesCreateRequest(**kwargs)