from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def _complete_matrix(self, operands: List[float]) -> List[float]:
    """Adds a, b, c, and d to an "e/f only" operand set (e.g Td)"""
    if len(operands) == 2:
        operands = [1.0, 0.0, 0.0, 1.0, *operands]
    return operands