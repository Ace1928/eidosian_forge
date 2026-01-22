import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def _is_valid_adls_put_header(header_name):
    """
    Returns:
        True if the specified header name is a valid header for the ADLS Put operation, False
        otherwise. For a list of valid headers, see
        https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/create
    """
    return header_name in {'Cache-Control', 'Content-Encoding', 'Content-Language', 'Content-Disposition', 'x-ms-cache-control', 'x-ms-content-type', 'x-ms-content-encoding', 'x-ms-content-language', 'x-ms-content-disposition', 'x-ms-rename-source', 'x-ms-lease-id', 'x-ms-properties', 'x-ms-permissions', 'x-ms-umask', 'x-ms-owner', 'x-ms-group', 'x-ms-acl', 'x-ms-proposed-lease-id', 'x-ms-expiry-option', 'x-ms-expiry-time', 'If-Match', 'If-None-Match', 'If-Modified-Since', 'If-Unmodified-Since', 'x-ms-source-if-match', 'x-ms-source-if-none-match', 'x-ms-source-if-modified-since', 'x-ms-source-if-unmodified-since', 'x-ms-encryption-key', 'x-ms-encryption-key-sha256', 'x-ms-encryption-algorithm', 'x-ms-encryption-context', 'x-ms-client-request-id', 'x-ms-date', 'x-ms-version'}