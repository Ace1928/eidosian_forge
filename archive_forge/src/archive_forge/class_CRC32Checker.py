import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
class CRC32Checker(BaseChecker):

    def __init__(self, header):
        self._header_name = header

    def _check_response(self, attempt_number, response):
        http_response = response[0]
        expected_crc = http_response.headers.get(self._header_name)
        if expected_crc is None:
            logger.debug('crc32 check skipped, the %s header is not in the http response.', self._header_name)
        else:
            actual_crc32 = crc32(response[0].content) & 4294967295
            if not actual_crc32 == int(expected_crc):
                logger.debug('retry needed: crc32 check failed, expected != actual: %s != %s', int(expected_crc), actual_crc32)
                raise ChecksumError(checksum_type='crc32', expected_checksum=int(expected_crc), actual_checksum=actual_crc32)