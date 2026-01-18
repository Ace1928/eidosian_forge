from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@property
def total_predict_batches_current_dataloader(self) -> Union[int, float]:
    """The total number of prediction batches, which may change from epoch to epoch for current dataloader.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the predict dataloader
        is of infinite size.

        """
    assert self._current_eval_dataloader_idx is not None
    return self.trainer.num_predict_batches[self._current_eval_dataloader_idx]