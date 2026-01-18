from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
    """Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors wrapped in a custom data
        structure.

        The data types listed below (and any arbitrary nesting of them) are supported out of the box:

        - :class:`torch.Tensor` or anything that implements `.to(...)`
        - :class:`list`
        - :class:`dict`
        - :class:`tuple`

        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

        Note:
            This hook should only transfer the data and not modify it, nor should it move the data to
            any other device than the one passed in as argument (unless you know what you are doing).
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.

        Example::

            def transfer_batch_to_device(self, batch, device, dataloader_idx):
                if isinstance(batch, CustomBatch):
                    # move all tensors in your custom data structure to the device
                    batch.samples = batch.samples.to(device)
                    batch.targets = batch.targets.to(device)
                elif dataloader_idx == 0:
                    # skip device transfer for the first dataloader or anything you wish
                    pass
                else:
                    batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
                return batch

        Raises:
            MisconfigurationException:
                If using IPUs, ``Trainer(accelerator='ipu')``.

        See Also:
            - :meth:`move_data_to_device`
            - :meth:`apply_to_collection`

        """
    return move_data_to_device(batch, device)