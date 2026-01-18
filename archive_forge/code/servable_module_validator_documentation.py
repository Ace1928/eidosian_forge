import contextlib
import logging
import time
from multiprocessing import Process
from typing import Any, Dict, Literal, Optional
import requests
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.serve.servable_module import ServableModule
from pytorch_lightning.strategies import DeepSpeedStrategy, FSDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_only
This method starts a server with a serve and ping endpoints.