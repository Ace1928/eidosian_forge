import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
def verify_splits(expected_splits: Optional[dict], recorded_splits: dict):
    if expected_splits is None:
        logger.info('Unable to verify splits sizes.')
        return
    if len(set(expected_splits) - set(recorded_splits)) > 0:
        raise ExpectedMoreSplits(str(set(expected_splits) - set(recorded_splits)))
    if len(set(recorded_splits) - set(expected_splits)) > 0:
        raise UnexpectedSplits(str(set(recorded_splits) - set(expected_splits)))
    bad_splits = [{'expected': expected_splits[name], 'recorded': recorded_splits[name]} for name in expected_splits if expected_splits[name].num_examples != recorded_splits[name].num_examples]
    if len(bad_splits) > 0:
        raise NonMatchingSplitsSizesError(str(bad_splits))
    logger.info('All the splits matched successfully.')