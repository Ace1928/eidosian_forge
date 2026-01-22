import warnings
from pathlib import Path
from typing import Optional, Type, Union
from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
class PossibleUserWarning(UserWarning):
    """Warnings that could be false positives."""