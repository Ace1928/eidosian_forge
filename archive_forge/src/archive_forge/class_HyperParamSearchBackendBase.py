from .integrations import (
from .trainer_utils import (
from .utils import logging
class HyperParamSearchBackendBase:
    name: str
    pip_package: str = None

    @staticmethod
    def is_available():
        raise NotImplementedError

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        raise NotImplementedError

    def default_hp_space(self, trial):
        raise NotImplementedError

    def ensure_available(self):
        if not self.is_available():
            raise RuntimeError(f'You picked the {self.name} backend, but it is not installed. Run {self.pip_install()}.')

    @classmethod
    def pip_install(cls):
        return f'`pip install {cls.pip_package or cls.name}`'