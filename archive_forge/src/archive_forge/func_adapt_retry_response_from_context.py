import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def adapt_retry_response_from_context(self, context):
    """Modify response back to user back from context."""
    metadata = context.get_retry_metadata()
    if context.parsed_response is not None:
        context.parsed_response.setdefault('ResponseMetadata', {}).update(metadata)