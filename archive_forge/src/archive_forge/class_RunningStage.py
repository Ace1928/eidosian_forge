from dataclasses import dataclass
from typing import Optional
from pytorch_lightning.utilities.enums import LightningEnum
class RunningStage(LightningEnum):
    """Enum for the current running stage.

    This stage complements :class:`TrainerFn` by specifying the current running stage for each function.
    More than one running stage value can be set while a :class:`TrainerFn` is running:

        - ``TrainerFn.FITTING`` - ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``
        - ``TrainerFn.VALIDATING`` - ``RunningStage.VALIDATING``
        - ``TrainerFn.TESTING`` - ``RunningStage.TESTING``
        - ``TrainerFn.PREDICTING`` - ``RunningStage.PREDICTING``

    """
    TRAINING = 'train'
    SANITY_CHECKING = 'sanity_check'
    VALIDATING = 'validate'
    TESTING = 'test'
    PREDICTING = 'predict'

    @property
    def evaluating(self) -> bool:
        return self in (self.VALIDATING, self.TESTING, self.SANITY_CHECKING)

    @property
    def dataloader_prefix(self) -> Optional[str]:
        if self in (self.VALIDATING, self.SANITY_CHECKING):
            return 'val'
        return self.value