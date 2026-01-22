import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def _is_valid_put_block_header(header_name):
    """
    Returns:
        True if the specified header name is a valid header for the Put Block operation, False
        otherwise. For a list of valid headers, see
        https://docs.microsoft.com/en-us/rest/api/storageservices/put-block#request-headers and
        https://docs.microsoft.com/en-us/rest/api/storageservices/put-block#
        request-headers-customer-provided-encryption-keys.
    """
    return header_name in {'Authorization', 'x-ms-date', 'x-ms-version', 'Content-Length', 'Content-MD5', 'x-ms-content-crc64', 'x-ms-encryption-scope', 'x-ms-lease-id', 'x-ms-client-request-id', 'x-ms-encryption-key', 'x-ms-encryption-key-sha256', 'x-ms-encryption-algorithm'}