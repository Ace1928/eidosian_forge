from dataclasses import dataclass
from typing import Optional
from pytorch_lightning.utilities.enums import LightningEnum
class TrainerFn(LightningEnum):
    """Enum for the user-facing functions of the :class:`~pytorch_lightning.trainer.trainer.Trainer` such as
    :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit` and
    :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`."""
    FITTING = 'fit'
    VALIDATING = 'validate'
    TESTING = 'test'
    PREDICTING = 'predict'