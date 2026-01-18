from typing import Any, Iterable, Optional, Union
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
from lightning_fabric.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger
from pytorch_lightning.trainer.connectors.logger_connector.result import _METRICS, _OUT_DICT, _PBAR_DICT
from pytorch_lightning.utilities.rank_zero import WarningCache
This function returns either batch or epoch metrics.