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
class BaseCRTRequestSerializer:

    def serialize_http_request(self, transfer_type, future):
        """Serialize CRT HTTP requests.

        :type transfer_type: string
        :param transfer_type: the type of transfer made,
            e.g 'put_object', 'get_object', 'delete_object'

        :type future: s3transfer.crt.CRTTransferFuture

        :rtype: awscrt.http.HttpRequest
        :returns: An unsigned HTTP request to be used for the CRT S3 client
        """
        raise NotImplementedError('serialize_http_request()')

    def translate_crt_exception(self, exception):
        raise NotImplementedError('translate_crt_exception()')