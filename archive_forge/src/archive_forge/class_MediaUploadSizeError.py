from __future__ import absolute_import
import json
import six
from googleapiclient import _helpers as util
class MediaUploadSizeError(Error):
    """Media is larger than the method can accept."""
    pass