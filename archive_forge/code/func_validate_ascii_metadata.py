import base64
import copy
import logging
import os
import re
import uuid
import warnings
from io import BytesIO
import botocore
import botocore.auth
from botocore import utils
from botocore.compat import (
from botocore.docs.utils import (
from botocore.endpoint_provider import VALID_HOST_LABEL_RE
from botocore.exceptions import (
from botocore.regions import EndpointResolverBuiltins
from botocore.signers import (
from botocore.utils import (
from botocore import retryhandler  # noqa
from botocore import translate  # noqa
from botocore.compat import MD5_AVAILABLE  # noqa
from botocore.exceptions import MissingServiceIdError  # noqa
from botocore.utils import hyphenize_service_id  # noqa
from botocore.utils import is_global_accesspoint  # noqa
from botocore.utils import SERVICE_NAME_ALIASES  # noqa
def validate_ascii_metadata(params, **kwargs):
    """Verify S3 Metadata only contains ascii characters.

    From: http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingMetadata.html

    "Amazon S3 stores user-defined metadata in lowercase. Each name, value pair
    must conform to US-ASCII when using REST and UTF-8 when using SOAP or
    browser-based uploads via POST."

    """
    metadata = params.get('Metadata')
    if not metadata or not isinstance(metadata, dict):
        return
    for key, value in metadata.items():
        try:
            key.encode('ascii')
            value.encode('ascii')
        except UnicodeEncodeError:
            error_msg = 'Non ascii characters found in S3 metadata for key "%s", value: "%s".  \nS3 metadata can only contain ASCII characters. ' % (key, value)
            raise ParamValidationError(report=error_msg)