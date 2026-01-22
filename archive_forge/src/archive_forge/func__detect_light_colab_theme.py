import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _detect_light_colab_theme() -> bool:
    """Detect if it's light theme in Colab."""
    try:
        import get_ipython
    except (NameError, ModuleNotFoundError):
        return False
    ipython = get_ipython()
    if 'google.colab' in str(ipython.__class__):
        try:
            from google.colab import output
            return output.eval_js('document.documentElement.matches("[theme=light]")')
        except ModuleNotFoundError:
            return False
    return False