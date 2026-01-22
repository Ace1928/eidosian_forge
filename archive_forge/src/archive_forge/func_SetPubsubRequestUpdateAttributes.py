from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def SetPubsubRequestUpdateAttributes(unused_job_ref, args, update_job_req):
    """Modify the Pubsub update request to update, remove, or clear attributes."""
    attributes = None
    if args.clear_attributes:
        attributes = {}
    elif args.update_attributes or args.remove_attributes:
        if args.update_attributes:
            attributes = args.update_attributes
        if args.remove_attributes:
            for key in args.remove_attributes:
                attributes[key] = None
    if attributes:
        update_job_req.job.pubsubTarget.attributes = _GenerateAdditionalProperties(attributes)
    return update_job_req