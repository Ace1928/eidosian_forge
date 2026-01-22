from typing import List, Union
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
class CodeInterpreterOutputImageImage(BaseModel):
    file_id: str
    '\n    The [file](https://platform.openai.com/docs/api-reference/files) ID of the\n    image.\n    '