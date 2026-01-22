import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def _extract_retryable_exception(config):
    applies_when = config['applies_when']
    if 'crc32body' in applies_when.get('response', {}):
        return [ChecksumError]
    elif 'socket_errors' in applies_when:
        exceptions = []
        for name in applies_when['socket_errors']:
            exceptions.extend(EXCEPTION_MAP[name])
        return exceptions