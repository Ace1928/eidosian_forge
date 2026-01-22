from .integrations import (
from .trainer_utils import (
from .utils import logging
class WandbBackend(HyperParamSearchBackendBase):
    name = 'wandb'

    @staticmethod
    def is_available():
        return is_wandb_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_wandb(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        return default_hp_space_wandb(trial)