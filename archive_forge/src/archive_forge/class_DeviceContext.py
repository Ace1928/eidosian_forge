import threading
from typing import Any, Dict
import torch._C._lazy
class DeviceContext:
    _CONTEXTS: Dict[str, Any] = dict()
    _CONTEXTS_LOCK = threading.Lock()

    def __init__(self, device):
        self.device = device