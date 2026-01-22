import logging
import warnings
import smart_open.bytebuffer
import smart_open.utils
from smart_open import constants
def Writer(bucket, blob, min_part_size=None, client=None, blob_properties=None, blob_open_kwargs=None):
    if blob_open_kwargs is None:
        blob_open_kwargs = {}
    if blob_properties is None:
        blob_properties = {}
    if client is None:
        client = google.cloud.storage.Client()
    blob_open_kwargs = {**_DEFAULT_WRITE_OPEN_KWARGS, **blob_open_kwargs}
    g_bucket = client.bucket(bucket)
    if not g_bucket.exists():
        raise google.cloud.exceptions.NotFound(f'bucket {bucket} not found')
    g_blob = g_bucket.blob(blob, chunk_size=min_part_size)
    for k, v in blob_properties.items():
        setattr(g_blob, k, v)
    _blob = g_blob.open('wb', **blob_open_kwargs)
    _blob.terminate = lambda: None
    return _blob