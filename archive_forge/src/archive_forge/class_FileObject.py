from typing import Optional
from typing_extensions import Literal
from .._models import BaseModel
class FileObject(BaseModel):
    id: str
    'The file identifier, which can be referenced in the API endpoints.'
    bytes: int
    'The size of the file, in bytes.'
    created_at: int
    'The Unix timestamp (in seconds) for when the file was created.'
    filename: str
    'The name of the file.'
    object: Literal['file']
    'The object type, which is always `file`.'
    purpose: Literal['fine-tune', 'fine-tune-results', 'assistants', 'assistants_output']
    'The intended purpose of the file.\n\n    Supported values are `fine-tune`, `fine-tune-results`, `assistants`, and\n    `assistants_output`.\n    '
    status: Literal['uploaded', 'processed', 'error']
    'Deprecated.\n\n    The current status of the file, which can be either `uploaded`, `processed`, or\n    `error`.\n    '
    status_details: Optional[str] = None
    'Deprecated.\n\n    For details on why a fine-tuning training file failed validation, see the\n    `error` field on `fine_tuning.job`.\n    '