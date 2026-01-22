import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
@staticmethod
def _parse_error(message, strict):
    if strict:
        raise ChangelogParseError(message)
    logger.warning(message)