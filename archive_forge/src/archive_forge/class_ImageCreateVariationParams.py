from __future__ import annotations
from typing import Union, Optional
from typing_extensions import Literal, Required, TypedDict
from .._types import FileTypes
class ImageCreateVariationParams(TypedDict, total=False):
    image: Required[FileTypes]
    'The image to use as the basis for the variation(s).\n\n    Must be a valid PNG file, less than 4MB, and square.\n    '
    model: Union[str, Literal['dall-e-2'], None]
    'The model to use for image generation.\n\n    Only `dall-e-2` is supported at this time.\n    '
    n: Optional[int]
    'The number of images to generate.\n\n    Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported.\n    '
    response_format: Optional[Literal['url', 'b64_json']]
    'The format in which the generated images are returned.\n\n    Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the\n    image has been generated.\n    '
    size: Optional[Literal['256x256', '512x512', '1024x1024']]
    'The size of the generated images.\n\n    Must be one of `256x256`, `512x512`, or `1024x1024`.\n    '
    user: str
    '\n    A unique identifier representing your end-user, which can help OpenAI to monitor\n    and detect abuse.\n    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).\n    '