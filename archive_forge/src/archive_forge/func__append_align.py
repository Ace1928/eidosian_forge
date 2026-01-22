from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_align(params: List[str], align: Optional[Union[int, str]]):
    """Append ALIGN property to params."""
    if align is not None:
        params.extend(['ALIGN', align])