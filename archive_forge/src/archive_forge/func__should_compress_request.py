import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
def _should_compress_request(config, request_dict, operation_model):
    if config.disable_request_compression is not True and config.signature_version != 'v2' and (operation_model.request_compression is not None):
        if not _is_compressible_type(request_dict):
            body_type = type(request_dict['body'])
            log_msg = 'Body type %s does not support compression.'
            logger.debug(log_msg, body_type)
            return False
        if operation_model.has_streaming_input:
            streaming_input = operation_model.get_streaming_input()
            streaming_metadata = streaming_input.metadata
            return 'requiresLength' not in streaming_metadata
        body_size = _get_body_size(request_dict['body'])
        min_size = config.request_min_compression_size_bytes
        return min_size <= body_size
    return False