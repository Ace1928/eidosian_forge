import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def _is_valid_adls_patch_header(header_name):
    """
    Returns:
        True if the specified header name is a valid header for the ADLS Patch operation, False
        otherwise. For a list of valid headers, see
        https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/update
    """
    return header_name in {'Content-Length', 'Content-MD5', 'x-ms-lease-id', 'x-ms-cache-control', 'x-ms-content-type', 'x-ms-content-disposition', 'x-ms-content-encoding', 'x-ms-content-language', 'x-ms-content-md5', 'x-ms-properties', 'x-ms-owner', 'x-ms-group', 'x-ms-permissions', 'x-ms-acl', 'If-Match', 'If-None-Match', 'If-Modified-Since', 'If-Unmodified-Since', 'x-ms-encryption-key', 'x-ms-encryption-key-sha256', 'x-ms-encryption-algorithm', 'x-ms-encryption-context', 'x-ms-client-request-id', 'x-ms-date', 'x-ms-version'}