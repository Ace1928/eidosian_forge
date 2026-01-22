import logging
from typing import Any, Dict, List, Optional, Union
import torch
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.data import _set_sampler_epoch, sized_len
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.fetchers import _DataFetcher
from pytorch_lightning.loops.progress import _Progress
from pytorch_lightning.loops.training_epoch_loop import _TrainingEpochLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached, _select_data_fetcher
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import _SUPPORTED_MODES, CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException, SIGTERMException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_warn
def _should_accumulate(self) -> bool:
    """Whether the gradients should be accumulated."""
    return self.epoch_loop._should_accumulate()