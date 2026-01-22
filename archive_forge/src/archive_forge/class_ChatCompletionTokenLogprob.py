from typing import List, Optional
from ..._models import BaseModel
class ChatCompletionTokenLogprob(BaseModel):
    token: str
    'The token.'
    bytes: Optional[List[int]] = None
    'A list of integers representing the UTF-8 bytes representation of the token.\n\n    Useful in instances where characters are represented by multiple tokens and\n    their byte representations must be combined to generate the correct text\n    representation. Can be `null` if there is no bytes representation for the token.\n    '
    logprob: float
    'The log probability of this token, if it is within the top 20 most likely\n    tokens.\n\n    Otherwise, the value `-9999.0` is used to signify that the token is very\n    unlikely.\n    '
    top_logprobs: List[TopLogprob]
    'List of the most likely tokens and their log probability, at this token\n    position.\n\n    In rare cases, there may be fewer than the number of requested `top_logprobs`\n    returned.\n    '