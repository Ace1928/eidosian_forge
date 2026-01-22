import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
class _ClientWrapper:
    """Wraps a client to inject the appropriate keyword args into each method call.

    The keyword args are a dictionary keyed by the fully qualified method name.
    For example, S3.Client.create_multipart_upload.

    See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#client

    This wrapper behaves identically to the client otherwise.
    """

    def __init__(self, client, kwargs):
        self.client = client
        self.kwargs = kwargs

    def __getattr__(self, method_name):
        method = getattr(self.client, method_name)
        kwargs = self.kwargs.get('S3.Client.%s' % method_name, {})
        return functools.partial(method, **kwargs)