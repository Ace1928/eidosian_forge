from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
class RendezvousTimeoutError(RendezvousError):
    """Raised when a rendezvous did not complete on time."""