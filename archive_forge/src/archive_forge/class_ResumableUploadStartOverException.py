from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ResumableUploadStartOverException(RetryableServiceException):
    """Exception raised for res. uploads that can be retried w/ new upload ID."""