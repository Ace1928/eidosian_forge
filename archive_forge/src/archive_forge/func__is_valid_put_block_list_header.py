import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def _is_valid_put_block_list_header(header_name):
    """
    Returns:
        True if the specified header name is a valid header for the Put Block List operation,
        False otherwise. For a list of valid headers, see https://docs.microsoft.com/en-us/
        rest/api/storageservices/put-block-list#request-headers and https://docs.microsoft.com/
        en-us/rest/api/storageservices/
        specifying-conditional-headers-for-blob-service-operations#Subheading1.
    """
    return header_name.startswith('x-ms-meta-') or header_name in {'Authorization', 'Date', 'x-ms-date', 'x-ms-version', 'Content-Length', 'Content-MD5', 'x-ms-content-crc64', 'x-ms-blob-cache-control', 'x-ms-blob-content-type', 'x-ms-blob-content-encoding', 'x-ms-blob-content-language', 'x-ms-blob-content-md5', 'x-ms-encryption-scope', 'x-ms-tags', 'x-ms-lease-id', 'x-ms-client-request-id', 'x-ms-blob-content-disposition', 'x-ms-access-tier', 'If-Modified-Since', 'If-Unmodified-Since', 'If-Match', 'If-None-Match'}