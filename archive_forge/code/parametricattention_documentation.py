from typing import Callable, Optional, Tuple
from ..config import registry
from ..model import Model
from ..types import Ragged
from ..util import get_width
Weight inputs by similarity to a learned vector