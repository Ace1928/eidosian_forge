import logging
import os
import pickle
import sys
import threading
import warnings
from types import ModuleType, TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type
from packaging.version import Version
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.utilities.migration.migration import _migration_index
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
class _FaultTolerantMode(LightningEnum):
    DISABLED = 'disabled'
    AUTOMATIC = 'automatic'
    MANUAL = 'manual'