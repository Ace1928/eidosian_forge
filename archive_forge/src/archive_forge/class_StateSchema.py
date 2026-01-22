import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
class StateSchema(ABC):
    """Schema class for Ray resource abstraction.

    The child class must be dataclass. All child classes
    - perform runtime type checking upon initialization.
    - are supposed to use `state_column` instead of `field`.
        It will allow the class to return filterable/detail columns.
        If `state_column` is not specified, that column is not filterable
        and for non-detail output.

    For example,
    ```
    @dataclass
    class State(StateSchema):
        column_a: str
        column_b: int = state_column(detail=True, filterable=True)

    s = State(column_a="abc", b=1)
    # Returns {"column_b"}
    s.filterable_columns()
    # Returns {"column_a"}
    s.base_columns()
    # Returns {"column_a", "column_b"}
    s.columns()
    ```

    In addition, the schema also provides a humanify abstract method to
    convert the state object into something human readable, ready for printing.

    Subclasses should override this method, providing logic to convert its own fields
    to something human readable, packaged and returned in a dict.

    Each field that wants to be humanified should include a 'format_fn' key in its
    metadata dictionary.
    """

    @classmethod
    def humanify(cls, state: dict) -> dict:
        """Convert the given state object into something human readable."""
        for f in fields(cls):
            if f.metadata.get('format_fn') is not None and f.name in state and (state[f.name] is not None):
                try:
                    state[f.name] = f.metadata['format_fn'](state[f.name])
                except Exception as e:
                    logger.error(f'Failed to format {f.name}:{state[f.name]} with {e}')
        return state

    @classmethod
    def list_columns(cls, detail: bool=True) -> List[str]:
        """Return a list of columns."""
        cols = []
        for f in fields(cls):
            if detail:
                cols.append(f.name)
            elif not f.metadata.get('detail', False):
                cols.append(f.name)
        return cols

    @classmethod
    def columns(cls) -> Set[str]:
        """Return a set of all columns."""
        return set(cls.list_columns())

    @classmethod
    def filterable_columns(cls) -> Set[str]:
        """Return a list of filterable columns"""
        filterable = set()
        for f in fields(cls):
            if f.metadata.get('filterable', False):
                filterable.add(f.name)
        return filterable

    @classmethod
    def base_columns(cls) -> Set[str]:
        """Return a list of base columns.

        Base columns mean columns to return when detail == False.
        """
        return set(cls.list_columns(detail=False))

    @classmethod
    def detail_columns(cls) -> Set[str]:
        """Return a list of detail columns.

        Detail columns mean columns to return when detail == True.
        """
        return set(cls.list_columns(detail=True))

    def asdict(self):
        return asdict(self)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)