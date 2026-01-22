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
class RenameTempFileHandler:

    def __init__(self, coordinator, final_filename, temp_filename, osutil):
        self._coordinator = coordinator
        self._final_filename = final_filename
        self._temp_filename = temp_filename
        self._osutil = osutil

    def __call__(self, **kwargs):
        error = kwargs['error']
        if error:
            self._osutil.remove_file(self._temp_filename)
        else:
            try:
                self._osutil.rename_file(self._temp_filename, self._final_filename)
            except Exception as e:
                self._osutil.remove_file(self._temp_filename)
                self._coordinator.set_exception(e)