from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.indexes import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
Remove data points from the specified index.

  ## EXAMPLES

  To remove data points from an index `123`, run:

    $ {command} 123 --datapoint-ids=example1,example2
    --project=example --region=us-central1

  Or put datapoint ids in a JSON file and run:

    $ {command} 123 --datapoints-from-file=example.json
    --project=example --region=us-central1
  