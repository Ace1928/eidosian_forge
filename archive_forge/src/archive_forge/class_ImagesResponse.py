from typing import List
from .image import Image
from .._models import BaseModel
class ImagesResponse(BaseModel):
    created: int
    data: List[Image]