from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
def GetTempFileName(storage_url):
    """Returns temporary file name for uncompressed file."""
    return '%s_.gstmp' % storage_url.object_name