from typing import Dict, List, Optional
from typing_extensions import Literal
from .._models import BaseModel
class CompletionChoice(BaseModel):
    finish_reason: Literal['stop', 'length', 'content_filter']
    'The reason the model stopped generating tokens.\n\n    This will be `stop` if the model hit a natural stop point or a provided stop\n    sequence, `length` if the maximum number of tokens specified in the request was\n    reached, or `content_filter` if content was omitted due to a flag from our\n    content filters.\n    '
    index: int
    logprobs: Optional[Logprobs] = None
    text: str