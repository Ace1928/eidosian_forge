from typing import Optional
from typing_extensions import Literal
from ....._models import BaseModel
class Image(BaseModel):
    file_id: Optional[str] = None
    '\n    The [file](https://platform.openai.com/docs/api-reference/files) ID of the\n    image.\n    '