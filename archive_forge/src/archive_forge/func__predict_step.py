from collections import OrderedDict
from typing import Any, Iterator, List, Optional, Union
import torch
from lightning_utilities import WarningCache
import pytorch_lightning as pl
from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.progress import _Progress
from pytorch_lightning.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from pytorch_lightning.overrides.distributed import _IndexBatchSamplerWrapper
from pytorch_lightning.strategies.launchers import _MultiProcessingLauncher
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT
def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int, dataloader_iter: Optional[Iterator]) -> None:
    """Runs the actual predict step together with all the necessary bookkeeping and the hooks tied to it.

        Args:
            batch: the current batch to run the prediction on
            batch_idx: The index of the current batch.
            dataloader_idx: the index of the dataloader producing the current batch.
            dataloader_iter: The iterator if using this step flavor.

        """
    trainer = self.trainer
    data_fetcher = self._data_fetcher
    assert data_fetcher is not None
    if not (using_dataloader_iter := isinstance(data_fetcher, _DataLoaderIterDataFetcher)):
        batch = trainer.precision_plugin.convert_input(batch)
        batch = trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = call._call_strategy_hook(trainer, 'batch_to_device', batch, dataloader_idx=dataloader_idx)
    self.batch_progress.increment_ready()
    if not using_dataloader_iter:
        any_on_epoch = self._store_data_for_prediction_writer(batch_idx, dataloader_idx)
    hook_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self.num_dataloaders > 1 else None)
    call._call_callback_hooks(trainer, 'on_predict_batch_start', *hook_kwargs.values())
    call._call_lightning_module_hook(trainer, 'on_predict_batch_start', *hook_kwargs.values())
    self.batch_progress.increment_started()
    step_args = self._build_step_args_from_hook_kwargs(hook_kwargs, 'predict_step') if not using_dataloader_iter else (dataloader_iter,)
    predictions = call._call_strategy_hook(trainer, 'predict_step', *step_args)
    if predictions is None:
        self._warning_cache.warn('predict returned None if it was on purpose, ignore this warning...')
    self.batch_progress.increment_processed()
    if using_dataloader_iter:
        batch = data_fetcher._batch
        batch_idx = data_fetcher._batch_idx
        dataloader_idx = data_fetcher._dataloader_idx
        hook_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self.num_dataloaders > 1 else None)
    call._call_callback_hooks(trainer, 'on_predict_batch_end', predictions, *hook_kwargs.values())
    call._call_lightning_module_hook(trainer, 'on_predict_batch_end', predictions, *hook_kwargs.values())
    self.batch_progress.increment_completed()
    if self._return_predictions or any_on_epoch:
        self._predictions[dataloader_idx].append(move_data_to_device(predictions, torch.device('cpu')))