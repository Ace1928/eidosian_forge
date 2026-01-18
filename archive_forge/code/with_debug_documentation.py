from typing import Any, Callable, Optional, Tuple, TypeVar
from ..model import Model
Debugging layer that wraps any layer and allows executing callbacks
    during the forward pass, backward pass and initialization. The callbacks
    will receive the same arguments as the functions they're called in.
    