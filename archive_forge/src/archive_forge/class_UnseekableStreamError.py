from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnseekableStreamError(BotoCoreError):
    """Need to seek a stream, but stream does not support seeking."""
    fmt = 'Need to rewind the stream {stream_object}, but stream is not seekable.'