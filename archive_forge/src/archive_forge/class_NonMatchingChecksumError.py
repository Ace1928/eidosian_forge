import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
class NonMatchingChecksumError(ChecksumVerificationException):
    """The downloaded file checksum don't match the expected checksum."""