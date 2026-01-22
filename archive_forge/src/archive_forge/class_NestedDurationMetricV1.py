from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
class NestedDurationMetricV1(BaseModel):
    """
    Nested Duration Metric Container

    {
        'operation_a': 240.0,
        'operation_b': 120.0,
    }
    """
    name: Optional[str] = 'nested_duration'
    data: Dict[str, DurationMetric] = Field(default_factory=dict, description='The nested duration metric data')

    @property
    def data_values(self) -> Dict[str, float]:
        """
        Returns the data values
        """
        return {k: v.total for k, v in self.data.items()}

    def items(self, sort: Optional[bool]=None):
        """
        Returns the dict_items view of the data
        """
        if sort:
            return dict(sorted(self.data.items(), key=lambda x: x[1].total, reverse=sort))
        return self.data_values.items()

    def __getitem__(self, key: str) -> DurationMetric:
        """
        Gets the value for the given key
        """
        if key not in self.data:
            self.data[key] = DurationMetric(name=key)
        return self.data[key]

    def __setitem__(self, key: str, value: DurationMetric):
        """
        Sets the value for the given key
        """
        self.data[key] = value

    def __getattr__(self, name: str) -> DurationMetric:
        """
        Gets the value for the given key
        """
        if name not in self.data:
            self.data[name] = DurationMetric(name=name)
        return self.data[name]

    def __setattr__(self, name: str, value: DurationMetric) -> None:
        """
        Sets the value for the given key
        """
        self.data[name] = value

    def __repr__(self) -> str:
        """
        Representation of the object
        """
        return f'{dict(self.items())}'

    def __str__(self) -> str:
        """
        Representation of the object
        """
        return self.__repr__()