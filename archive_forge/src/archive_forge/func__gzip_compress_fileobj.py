import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
def _gzip_compress_fileobj(body):
    compressed_obj = io.BytesIO()
    with GzipFile(fileobj=compressed_obj, mode='wb') as gz:
        while True:
            chunk = body.read(8192)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode('utf-8')
            gz.write(chunk)
    compressed_obj.seek(0)
    return compressed_obj