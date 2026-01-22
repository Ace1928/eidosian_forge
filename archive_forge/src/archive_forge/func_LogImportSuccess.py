from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def LogImportSuccess(response, args):
    path = args.source
    if not args.async_:
        if path != '-':
            log.status.Print('Successfully imported agent from [{}].'.format(path))
        else:
            log.status.Print('Successfully imported agent.')
        if args.replace_all:
            log.status.Print('Replaced all existing resources.')
    return response