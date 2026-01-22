import logging
import threading
from io import BytesIO
import awscrt.http
import awscrt.s3
import botocore.awsrequest
import botocore.session
from awscrt.auth import (
from awscrt.io import (
from awscrt.s3 import S3Client, S3RequestTlsMode, S3RequestType
from botocore import UNSIGNED
from botocore.compat import urlsplit
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from s3transfer.constants import MB
from s3transfer.exceptions import TransferNotDoneError
from s3transfer.futures import BaseTransferFuture, BaseTransferMeta
from s3transfer.utils import (
class S3ClientArgsCreator:

    def __init__(self, crt_request_serializer, os_utils):
        self._request_serializer = crt_request_serializer
        self._os_utils = os_utils

    def get_make_request_args(self, request_type, call_args, coordinator, future, on_done_after_calls):
        request_args_handler = getattr(self, f'_get_make_request_args_{request_type}', self._default_get_make_request_args)
        return request_args_handler(request_type=request_type, call_args=call_args, coordinator=coordinator, future=future, on_done_before_calls=[], on_done_after_calls=on_done_after_calls)

    def get_crt_callback(self, future, callback_type, before_subscribers=None, after_subscribers=None):

        def invoke_all_callbacks(*args, **kwargs):
            callbacks_list = []
            if before_subscribers is not None:
                callbacks_list += before_subscribers
            callbacks_list += get_callbacks(future, callback_type)
            if after_subscribers is not None:
                callbacks_list += after_subscribers
            for callback in callbacks_list:
                if callback_type == 'progress':
                    callback(bytes_transferred=args[0])
                else:
                    callback(*args, **kwargs)
        return invoke_all_callbacks

    def _get_make_request_args_put_object(self, request_type, call_args, coordinator, future, on_done_before_calls, on_done_after_calls):
        send_filepath = None
        if isinstance(call_args.fileobj, str):
            send_filepath = call_args.fileobj
            data_len = self._os_utils.get_file_size(send_filepath)
            call_args.extra_args['ContentLength'] = data_len
        else:
            call_args.extra_args['Body'] = call_args.fileobj
        checksum_algorithm = call_args.extra_args.pop('ChecksumAlgorithm', 'CRC32').upper()
        checksum_config = awscrt.s3.S3ChecksumConfig(algorithm=awscrt.s3.S3ChecksumAlgorithm[checksum_algorithm], location=awscrt.s3.S3ChecksumLocation.TRAILER)
        call_args.extra_args['ContentMD5'] = 'override-to-be-removed'
        make_request_args = self._default_get_make_request_args(request_type=request_type, call_args=call_args, coordinator=coordinator, future=future, on_done_before_calls=on_done_before_calls, on_done_after_calls=on_done_after_calls)
        make_request_args['send_filepath'] = send_filepath
        make_request_args['checksum_config'] = checksum_config
        return make_request_args

    def _get_make_request_args_get_object(self, request_type, call_args, coordinator, future, on_done_before_calls, on_done_after_calls):
        recv_filepath = None
        on_body = None
        checksum_config = awscrt.s3.S3ChecksumConfig(validate_response=True)
        if isinstance(call_args.fileobj, str):
            final_filepath = call_args.fileobj
            recv_filepath = self._os_utils.get_temp_filename(final_filepath)
            on_done_before_calls.append(RenameTempFileHandler(coordinator, final_filepath, recv_filepath, self._os_utils))
        else:
            on_body = OnBodyFileObjWriter(call_args.fileobj)
        make_request_args = self._default_get_make_request_args(request_type=request_type, call_args=call_args, coordinator=coordinator, future=future, on_done_before_calls=on_done_before_calls, on_done_after_calls=on_done_after_calls)
        make_request_args['recv_filepath'] = recv_filepath
        make_request_args['on_body'] = on_body
        make_request_args['checksum_config'] = checksum_config
        return make_request_args

    def _default_get_make_request_args(self, request_type, call_args, coordinator, future, on_done_before_calls, on_done_after_calls):
        make_request_args = {'request': self._request_serializer.serialize_http_request(request_type, future), 'type': getattr(S3RequestType, request_type.upper(), S3RequestType.DEFAULT), 'on_done': self.get_crt_callback(future, 'done', on_done_before_calls, on_done_after_calls), 'on_progress': self.get_crt_callback(future, 'progress')}
        if is_s3express_bucket(call_args.bucket):
            make_request_args['signing_config'] = AwsSigningConfig(algorithm=AwsSigningAlgorithm.V4_S3EXPRESS)
        return make_request_args