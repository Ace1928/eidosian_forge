from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
def _get_weaviate_datatype(self, value: Optional[Union[str, int, float, bool]]=None) -> Tuple[str, Union[str, int, float, bool]]:
    """
        Determines the type of the comparison value and converts it to RFC3339 format if it is as date,
        as Weaviate requires dates to be in RFC3339 format including the time and timezone.

        """
    if value is None:
        assert not isinstance(self.comparison_value, list)
        value = self.comparison_value
    if isinstance(value, str):
        try:
            value = convert_date_to_rfc3339(value)
            data_type = 'valueDate'
        except ValueError:
            data_type = 'valueText' if self.field_name == 'content' else 'valueString'
    elif isinstance(value, int):
        data_type = 'valueInt'
    elif isinstance(value, float):
        data_type = 'valueNumber'
    elif isinstance(value, bool):
        data_type = 'valueBoolean'
    else:
        raise ValueError(f'Unsupported data type of comparison value for {self.__class__.__name__}.Value needs to be of type str, int, float, or bool.')
    return (data_type, value)