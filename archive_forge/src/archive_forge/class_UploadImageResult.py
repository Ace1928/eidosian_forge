import importlib
import math
import re
from enum import Enum
class UploadImageResult(Enum):
    """
    Result of uploading an image.

    SUCCESS:        user successfully uploaded an image
    OBJECTIONABLE:  the image contains objectionable content
    ERROR:          there was an error
    """
    SUCCESS = 0
    OBJECTIONABLE = 1
    ERROR = 2