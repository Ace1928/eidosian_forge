from typing import Any, Callable
from torchmetrics.metric import Metric
def _wrap_update(self, update: Callable) -> Callable:
    """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
    return update