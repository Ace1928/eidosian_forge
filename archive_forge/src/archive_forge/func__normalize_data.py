from sentry_sdk import consts
from sentry_sdk._types import TYPE_CHECKING
import sentry_sdk
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import logger, capture_internal_exceptions, event_from_exception
def _normalize_data(data):
    if hasattr(data, 'model_dump'):
        try:
            return data.model_dump()
        except Exception as e:
            logger.warning('Could not convert pydantic data to JSON: %s', e)
            return data
    if isinstance(data, list):
        return list((_normalize_data(x) for x in data))
    if isinstance(data, dict):
        return {k: _normalize_data(v) for k, v in data.items()}
    return data