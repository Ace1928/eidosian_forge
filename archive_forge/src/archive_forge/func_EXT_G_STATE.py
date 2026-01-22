from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
@classmethod
@property
def EXT_G_STATE(cls) -> str:
    deprecate_with_replacement('Ressources', 'Resources', '5.0.0')
    return '/ExtGState'