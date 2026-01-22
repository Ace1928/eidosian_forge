from sentry_sdk import consts
from sentry_sdk._types import TYPE_CHECKING
import sentry_sdk
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import logger, capture_internal_exceptions, event_from_exception
class OpenAIIntegration(Integration):
    identifier = 'openai'

    def __init__(self, include_prompts=True):
        self.include_prompts = include_prompts

    @staticmethod
    def setup_once():
        Completions.create = _wrap_chat_completion_create(Completions.create)
        Embeddings.create = _wrap_embeddings_create(Embeddings.create)