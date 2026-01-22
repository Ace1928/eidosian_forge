import dataclasses
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
class ImageAttribute(FileAttribute):

    @staticmethod
    def get_file_name(attr_name: Optional[str]=None) -> str:
        return f'{attr_name}.png' if attr_name else 'image.png'