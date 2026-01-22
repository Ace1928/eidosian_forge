from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def SetOrderByFromSortBy(ref, args, request):
    """Set orderBy attribute on message from common --sort-by flag."""
    del ref
    if args.sort_by:
        order_by_fields = []
        for field in args.sort_by:
            if field.startswith('~'):
                field = field.lstrip('~') + ' desc'
            else:
                field += ' asc'
            order_by_fields.append(field)
        request.orderBy = ','.join(order_by_fields)
    return request