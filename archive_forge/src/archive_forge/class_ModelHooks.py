from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
class ModelHooks:
    """Hooks to be used in LightningModule."""

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.

        If on DDP it is called on every process

        """

    def on_fit_end(self) -> None:
        """Called at the very end of fit.

        If on DDP it is called on every process

        """

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""

    def on_train_end(self) -> None:
        """Called at the end of training before logger experiment is closed."""

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""

    def on_validation_end(self) -> None:
        """Called at the end of validation."""

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""

    def on_test_end(self) -> None:
        """Called at the end of testing."""

    def on_predict_start(self) -> None:
        """Called at the beginning of predicting."""

    def on_predict_end(self) -> None:
        """Called at the end of predicting."""

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        """Called in the training loop before anything happens for that batch.

        If you return -1 here, you will skip training for the rest of the current epoch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch

        """

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called in the training loop after the batch.

        Args:
            outputs: The outputs of training_step(x)
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch

        """

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
        """Called in the validation loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the validation DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
        """Called in the validation loop after the batch.

        Args:
            outputs: The outputs of validation_step(x)
            batch: The batched data as it is returned by the validation DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
        """Called in the test loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
        """Called in the test loop after the batch.

        Args:
            outputs: The outputs of test_step(x)
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
        """Called in the predict loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the test DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
        """Called in the predict loop after the batch.

        Args:
            outputs: The outputs of predict_step(x)
            batch: The batched data as it is returned by the prediction DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

    def on_validation_model_zero_grad(self) -> None:
        """Called by the training loop to release gradients before entering the validation loop."""
        zero_grad_kwargs = {} if _TORCH_GREATER_EQUAL_2_0 else {'set_to_none': True}
        self.zero_grad(**zero_grad_kwargs)

    def on_validation_model_eval(self) -> None:
        """Called when the validation loop starts.

        The validation loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
        to change the behavior. See also :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_validation_model_train`.

        """
        self.trainer.model.eval()

    def on_validation_model_train(self) -> None:
        """Called when the validation loop ends.

        The validation loop by default restores the `training` mode of the LightningModule to what it was before
        starting validation. Override this hook to change the behavior. See also
        :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_validation_model_eval`.

        """
        self.trainer.model.train()

    def on_test_model_eval(self) -> None:
        """Called when the test loop starts.

        The test loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
        to change the behavior. See also :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_test_model_train`.

        """
        self.trainer.model.eval()

    def on_test_model_train(self) -> None:
        """Called when the test loop ends.

        The test loop by default restores the `training` mode of the LightningModule to what it was before
        starting testing. Override this hook to change the behavior. See also
        :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_test_model_eval`.

        """
        self.trainer.model.train()

    def on_predict_model_eval(self) -> None:
        """Called when the predict loop starts.

        The predict loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
        to change the behavior.

        """
        self.trainer.model.eval()

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch.

        To access all batch outputs at the end of the epoch, you can cache step outputs as an attribute of the
        :class:`~pytorch_lightning.LightningModule` and access them in this hook:

        .. code-block:: python

            class MyLightningModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.training_step_outputs = []

                def training_step(self):
                    loss = ...
                    self.training_step_outputs.append(loss)
                    return loss

                def on_train_epoch_end(self):
                    # do something with all training_step outputs, for example:
                    epoch_mean = torch.stack(self.training_step_outputs).mean()
                    self.log("training_epoch_mean", epoch_mean)
                    # free up the memory
                    self.training_step_outputs.clear()

        """

    def on_validation_epoch_start(self) -> None:
        """Called in the validation loop at the very beginning of the epoch."""

    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""

    def on_test_epoch_start(self) -> None:
        """Called in the test loop at the very beginning of the epoch."""

    def on_test_epoch_end(self) -> None:
        """Called in the test loop at the very end of the epoch."""

    def on_predict_epoch_start(self) -> None:
        """Called at the beginning of predicting."""

    def on_predict_epoch_end(self) -> None:
        """Called at the end of predicting."""

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """Called after ``training_step()`` and before ``optimizer.zero_grad()``.

        Called in the training loop after taking an optimizer step and before zeroing grads.
        Good place to inspect weight information with weights updated.

        This is where it is called::

            for optimizer in optimizers:
                out = training_step(...)

                model.on_before_zero_grad(optimizer) # < ---- called here
                optimizer.zero_grad()

                backward()

        Args:
            optimizer: The optimizer for which grads should be zeroed.

        """

    def on_before_backward(self, loss: Tensor) -> None:
        """Called before ``loss.backward()``.

        Args:
            loss: Loss divided by number of batches for gradient accumulation and scaled if using AMP.

        """
        pass

    def on_after_backward(self) -> None:
        """Called after ``loss.backward()`` and before optimizers are stepped.

        Note:
            If using native AMP, the gradients will not be unscaled at this point.
            Use the ``on_before_optimizer_step`` if you need the unscaled gradients.

        """

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Called before ``optimizer.step()``.

        If using gradient accumulation, the hook is called once the gradients have been accumulated.
        See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.accumulate_grad_batches`.

        If using AMP, the loss will be unscaled before calling this hook.
        See these `docs <https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients>`__
        for more information on the scaling of gradients.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            optimizer: Current optimizer being used.

        Example::

            def on_before_optimizer_step(self, optimizer):
                # example to inspect gradient information in tensorboard
                if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
                    for k, v in self.named_parameters():
                        self.logger.experiment.add_histogram(
                            tag=k, values=v.grad, global_step=self.trainer.global_step
                        )

        """

    def configure_sharded_model(self) -> None:
        """Deprecated.

        Use :meth:`~pytorch_lightning.core.hooks.ModelHooks.configure_model` instead.

        """

    def configure_model(self) -> None:
        """Hook to create modules in a strategy and precision aware context.

        This is particularly useful for when using sharded strategies (FSDP and DeepSpeed), where we'd like to shard
        the model instantly to save memory and initialization time.
        For non-sharded strategies, you can choose to override this hook or to initialize your model under the
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.init_module` context manager.

        This hook is called during each of fit/val/test/predict stages in the same process, so ensure that
        implementation of this hook is **idempotent**, i.e., after the first time the hook is called, subsequent calls
        to it should be a no-op.

        """