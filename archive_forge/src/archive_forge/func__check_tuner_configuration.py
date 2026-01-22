from typing import TYPE_CHECKING, Literal, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
def _check_tuner_configuration(train_dataloaders: Optional[Union[TRAIN_DATALOADERS, 'pl.LightningDataModule']]=None, val_dataloaders: Optional[EVAL_DATALOADERS]=None, dataloaders: Optional[EVAL_DATALOADERS]=None, method: Literal['fit', 'validate', 'test', 'predict']='fit') -> None:
    supported_methods = ('fit', 'validate', 'test', 'predict')
    if method not in supported_methods:
        raise ValueError(f'method {method!r} is invalid. Should be one of {supported_methods}.')
    if method == 'fit':
        if dataloaders is not None:
            raise MisconfigurationException(f'In tuner with method={method!r}, `dataloaders` argument should be None, please consider setting `train_dataloaders` and `val_dataloaders` instead.')
    elif train_dataloaders is not None or val_dataloaders is not None:
        raise MisconfigurationException(f'In tuner with `method`={method!r}, `train_dataloaders` and `val_dataloaders` arguments should be None, please consider setting `dataloaders` instead.')