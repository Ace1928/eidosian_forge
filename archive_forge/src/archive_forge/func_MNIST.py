import logging
import os
import random
import time
import urllib
from typing import Any, Callable, Optional, Sized, Tuple, Union
from urllib.error import HTTPError
from warnings import warn
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from lightning_fabric.utilities.imports import _IS_WINDOWS
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
def MNIST(*args: Any, **kwargs: Any) -> Dataset:
    torchvision_mnist_available = not bool(os.getenv('PL_USE_MOCKED_MNIST', False))
    if torchvision_mnist_available:
        try:
            from torchvision.datasets import MNIST
            MNIST(_DATASETS_PATH, download=True)
        except HTTPError as ex:
            print(f'Error {ex} downloading `torchvision.datasets.MNIST`')
            torchvision_mnist_available = False
    if not torchvision_mnist_available:
        print('`torchvision.datasets.MNIST` not available. Using our hosted version')
        MNIST = _MNIST
    return MNIST(*args, **kwargs)