from dataclasses import dataclass
from typing import Optional
from pytorch_lightning.utilities.enums import LightningEnum
class TrainerStatus(LightningEnum):
    """Enum for the status of the :class:`~pytorch_lightning.trainer.trainer.Trainer`"""
    INITIALIZING = 'initializing'
    RUNNING = 'running'
    FINISHED = 'finished'
    INTERRUPTED = 'interrupted'

    @property
    def stopped(self) -> bool:
        return self in (self.FINISHED, self.INTERRUPTED)