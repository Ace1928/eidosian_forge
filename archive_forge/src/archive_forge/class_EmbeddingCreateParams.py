from __future__ import annotations
from typing import List, Union, Iterable
from typing_extensions import Literal, Required, TypedDict
class EmbeddingCreateParams(TypedDict, total=False):
    input: Required[Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]]
    'Input text to embed, encoded as a string or array of tokens.\n\n    To embed multiple inputs in a single request, pass an array of strings or array\n    of token arrays. The input must not exceed the max input tokens for the model\n    (8192 tokens for `text-embedding-ada-002`), cannot be an empty string, and any\n    array must be 2048 dimensions or less.\n    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)\n    for counting tokens.\n    '
    model: Required[Union[str, Literal['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']]]
    'ID of the model to use.\n\n    You can use the\n    [List models](https://platform.openai.com/docs/api-reference/models/list) API to\n    see all of your available models, or see our\n    [Model overview](https://platform.openai.com/docs/models/overview) for\n    descriptions of them.\n    '
    dimensions: int
    'The number of dimensions the resulting output embeddings should have.\n\n    Only supported in `text-embedding-3` and later models.\n    '
    encoding_format: Literal['float', 'base64']
    'The format to return the embeddings in.\n\n    Can be either `float` or [`base64`](https://pypi.org/project/pybase64/).\n    '
    user: str
    '\n    A unique identifier representing your end-user, which can help OpenAI to monitor\n    and detect abuse.\n    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).\n    '