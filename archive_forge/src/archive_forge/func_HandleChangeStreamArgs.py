from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def HandleChangeStreamArgs(unused_ref, args, req):
    if args.change_stream_retention_period:
        req.table.changeStreamConfig = CreateChangeStreamConfig(args.change_stream_retention_period)
    return req