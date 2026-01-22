from typing import Optional, Union
import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.profilers import (
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable, _habana_available_and_importable
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _determine_batch_limits(batches: Optional[Union[int, float]], name: str) -> Union[int, float]:
    if batches is None:
        return 1.0
    if isinstance(batches, int) and batches == 1:
        if name == 'limit_train_batches':
            message = '1 batch per epoch will be used.'
        elif name == 'val_check_interval':
            message = 'validation will run after every batch.'
        else:
            message = '1 batch will be used.'
        rank_zero_info(f'`Trainer({name}=1)` was configured so {message}')
    elif isinstance(batches, float) and batches == 1.0:
        if name == 'limit_train_batches':
            message = '100% of the batches per epoch will be used.'
        elif name == 'val_check_interval':
            message = 'validation will run at the end of the training epoch.'
        else:
            message = '100% of the batches will be used.'
        rank_zero_info(f'`Trainer({name}=1.0)` was configured so {message}.')
    if 0 <= batches <= 1:
        return batches
    if batches > 1 and batches % 1.0 == 0:
        return int(batches)
    raise MisconfigurationException(f'You have passed invalid value {batches} for {name}, it has to be in [0.0, 1.0] or an int.')