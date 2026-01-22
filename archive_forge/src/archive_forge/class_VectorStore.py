from typing import Optional
from typing_extensions import Literal
from ..._models import BaseModel
class VectorStore(BaseModel):
    id: str
    'The identifier, which can be referenced in API endpoints.'
    bytes: int
    'The byte size of the vector store.'
    created_at: int
    'The Unix timestamp (in seconds) for when the vector store was created.'
    file_counts: FileCounts
    last_active_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the vector store was last active.'
    metadata: Optional[object] = None
    'Set of 16 key-value pairs that can be attached to an object.\n\n    This can be useful for storing additional information about the object in a\n    structured format. Keys can be a maximum of 64 characters long and values can be\n    a maxium of 512 characters long.\n    '
    name: str
    'The name of the vector store.'
    object: Literal['vector_store']
    'The object type, which is always `vector_store`.'
    status: Literal['expired', 'in_progress', 'completed']
    '\n    The status of the vector store, which can be either `expired`, `in_progress`, or\n    `completed`. A status of `completed` indicates that the vector store is ready\n    for use.\n    '
    expires_after: Optional[ExpiresAfter] = None
    'The expiration policy for a vector store.'
    expires_at: Optional[int] = None
    'The Unix timestamp (in seconds) for when the vector store will expire.'