import inspect
from typing import IO, Any, Dict, Iterable, Optional, Union, cast
from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import Self
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pytorch_lightning.core.hooks import DataHooks
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.core.saving import _load_from_checkpoint
from pytorch_lightning.utilities.model_helpers import _restricted_classmethod
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
Primary way of loading a datamodule from a checkpoint. When Lightning saves a checkpoint it stores the
        arguments passed to ``__init__``  in the checkpoint under ``"datamodule_hyper_parameters"``.

        Any arguments specified through \*\*kwargs will override args stored in ``"datamodule_hyper_parameters"``.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                as in this example::

                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningDataModule` for use.

                If your datamodule's ``hparams`` argument is :class:`~argparse.Namespace`
                and ``.yaml`` file has hierarchical structure, you need to refactor your datamodule to treat
                ``hparams`` as :class:`~dict`.
            \**kwargs: Any extra keyword args needed to init the datamodule. Can also be used to override saved
                hyperparameter values.

        Return:
            :class:`LightningDataModule` instance with loaded weights and hyperparameters (if available).

        Note:
            ``load_from_checkpoint`` is a **class** method. You must use your :class:`LightningDataModule`
            **class** to call it instead of the :class:`LightningDataModule` instance, or a
            ``TypeError`` will be raised.

        Example::

            # load weights without mapping ...
            datamodule = MyLightningDataModule.load_from_checkpoint('path/to/checkpoint.ckpt')

            # or load weights and hyperparameters from separate files.
            datamodule = MyLightningDataModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                hparams_file='/path/to/hparams_file.yaml'
            )

            # override some of the params with new values
            datamodule = MyLightningDataModule.load_from_checkpoint(
                PATH,
                batch_size=32,
                num_workers=10,
            )

        