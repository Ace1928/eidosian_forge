import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
This function is responsible of sanitizing ``parameters_to_prune`` and ``parameter_names``. If
        ``parameters_to_prune is None``, it will be generated with all parameters of the model.

        Raises:
            MisconfigurationException:
                If ``parameters_to_prune`` doesn't exist in the model, or
                if ``parameters_to_prune`` is neither a list nor a tuple.

        