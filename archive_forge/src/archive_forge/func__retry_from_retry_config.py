import collections
import grpc
from google.api_core import exceptions
from google.api_core import retry
from google.api_core import timeout
def _retry_from_retry_config(retry_params, retry_codes, retry_impl=retry.Retry):
    """Creates a Retry object given a gapic retry configuration.

    DEPRECATED: instantiate retry and timeout classes directly instead.

    Args:
        retry_params (dict): The retry parameter values, for example::

            {
                "initial_retry_delay_millis": 1000,
                "retry_delay_multiplier": 2.5,
                "max_retry_delay_millis": 120000,
                "initial_rpc_timeout_millis": 120000,
                "rpc_timeout_multiplier": 1.0,
                "max_rpc_timeout_millis": 120000,
                "total_timeout_millis": 600000
            }

        retry_codes (sequence[str]): The list of retryable gRPC error code
            names.

    Returns:
        google.api_core.retry.Retry: The default retry object for the method.
    """
    exception_classes = [_exception_class_for_grpc_status_name(code) for code in retry_codes]
    return retry_impl(retry.if_exception_type(*exception_classes), initial=retry_params['initial_retry_delay_millis'] / _MILLIS_PER_SECOND, maximum=retry_params['max_retry_delay_millis'] / _MILLIS_PER_SECOND, multiplier=retry_params['retry_delay_multiplier'], deadline=retry_params['total_timeout_millis'] / _MILLIS_PER_SECOND)