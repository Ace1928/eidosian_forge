from typing import Any, Dict, Optional
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.accelerators.cpu import _PSUTIL_AVAILABLE
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _prefix_metric_keys(metrics_dict: Dict[str, float], prefix: str, separator: str) -> Dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}