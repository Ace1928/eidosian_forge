import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _Stateful
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning import loops  # import as loops to avoid circular imports
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.optimization import _AutomaticOptimization, _ManualOptimization
from pytorch_lightning.loops.optimization.automatic import _OUTPUTS_TYPE as _OPTIMIZER_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.optimization.manual import _OUTPUTS_TYPE as _MANUAL_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.progress import _BatchProgress, _SchedulerProgress
from pytorch_lightning.loops.utilities import _is_max_limit_reached
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException, SIGTERMException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def _on_before_fetch(self) -> None:
    self.trainer.profiler.start(f'[{self.__class__.__name__}].train_dataloader_next')