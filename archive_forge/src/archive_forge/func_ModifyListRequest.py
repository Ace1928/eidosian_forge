from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner.resource_args import CloudKmsKeyName
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.credentials import http
from googlecloudsdk.core.util import times
import six
from six.moves import http_client as httplib
from six.moves import urllib
def ModifyListRequest(instance_ref, args, req):
    """Parse arguments and construct list backups request."""
    req.parent = instance_ref.RelativeName()
    if args.database:
        database = instance_ref.RelativeName() + '/databases/' + args.database
        req.filter = 'database="{}"'.format(database)
    return req