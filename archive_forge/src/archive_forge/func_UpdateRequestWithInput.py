from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def UpdateRequestWithInput(unused_ref, args, request):
    """The Python hook for yaml commands to inject content into the request."""
    path = args.input_path
    if os.path.isfile(path):
        request.inputContent = files.ReadBinaryFileContents(path)
    elif storage_util.ObjectReference.IsStorageUrl(path):
        request.inputUri = path
    else:
        raise VideoUriFormatError(INPUT_ERROR_MESSAGE.format(path))
    return request