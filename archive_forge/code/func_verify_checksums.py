import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
def verify_checksums(expected_checksums: Optional[dict], recorded_checksums: dict, verification_name=None):
    if expected_checksums is None:
        logger.info('Unable to verify checksums.')
        return
    if len(set(expected_checksums) - set(recorded_checksums)) > 0:
        raise ExpectedMoreDownloadedFiles(str(set(expected_checksums) - set(recorded_checksums)))
    if len(set(recorded_checksums) - set(expected_checksums)) > 0:
        raise UnexpectedDownloadedFile(str(set(recorded_checksums) - set(expected_checksums)))
    bad_urls = [url for url in expected_checksums if expected_checksums[url] != recorded_checksums[url]]
    for_verification_name = ' for ' + verification_name if verification_name is not None else ''
    if len(bad_urls) > 0:
        raise NonMatchingChecksumError(f"Checksums didn't match{for_verification_name}:\n{bad_urls}\nSet `verification_mode='no_checks'` to skip checksums verification and ignore this error")
    logger.info('All the checksums matched successfully' + for_verification_name)