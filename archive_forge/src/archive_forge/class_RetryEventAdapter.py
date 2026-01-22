import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class RetryEventAdapter:
    """Adapter to existing retry interface used in the endpoints layer.

    This existing interface for determining if a retry needs to happen
    is event based and used in ``botocore.endpoint``.  The interface has
    grown organically over the years and could use some cleanup.  This
    adapter converts that interface into the interface used by the
    new retry strategies.

    """

    def create_retry_context(self, **kwargs):
        """Create context based on needs-retry kwargs."""
        response = kwargs['response']
        if response is None:
            http_response = None
            parsed_response = None
        else:
            http_response, parsed_response = response
        context = RetryContext(attempt_number=kwargs['attempts'], operation_model=kwargs['operation'], http_response=http_response, parsed_response=parsed_response, caught_exception=kwargs['caught_exception'], request_context=kwargs['request_dict']['context'])
        return context

    def adapt_retry_response_from_context(self, context):
        """Modify response back to user back from context."""
        metadata = context.get_retry_metadata()
        if context.parsed_response is not None:
            context.parsed_response.setdefault('ResponseMetadata', {}).update(metadata)