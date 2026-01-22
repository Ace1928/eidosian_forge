from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from .chat_completion_token_logprob import ChatCompletionTokenLogprob

    This fingerprint represents the backend configuration that the model runs with.
    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    