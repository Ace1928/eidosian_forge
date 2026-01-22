import logging
from typing import TypeVar
from ray.train._internal.utils import Singleton
from ray.train._internal.worker_group import WorkerGroup
from ray.util.annotations import DeveloperAPI
from ray.widgets import make_table_html_repr
@DeveloperAPI
class BackendConfig:
    """Parent class for configurations of training backend."""

    @property
    def backend_cls(self):
        return Backend

    def _repr_html_(self) -> str:
        return make_table_html_repr(obj=self, title=type(self).__name__)