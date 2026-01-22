from __future__ import absolute_import
import json
import six
from googleapiclient import _helpers as util
class ResumableUploadError(HttpError):
    """Error occurred during resumable upload."""
    pass