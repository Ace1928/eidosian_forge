from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
class RendezvousClosedError(RendezvousError):
    """Raised when a rendezvous is closed."""