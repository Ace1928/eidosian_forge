from __future__ import annotations
from typing import List
from typing_extensions import Required, TypedDict
class FileBatchCreateParams(TypedDict, total=False):
    file_ids: Required[List[str]]
    '\n    A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that\n    the vector store should use. Useful for tools like `file_search` that can access\n    files.\n    '