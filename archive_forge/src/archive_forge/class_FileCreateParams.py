from __future__ import annotations
from typing_extensions import Required, TypedDict
class FileCreateParams(TypedDict, total=False):
    file_id: Required[str]
    '\n    A [File](https://platform.openai.com/docs/api-reference/files) ID (with\n    `purpose="assistants"`) that the assistant should use. Useful for tools like\n    `retrieval` and `code_interpreter` that can access files.\n    '