from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class strip_label_details:
    """
    Strip Label Details
    """
    variables: dict[str, str]
    meta: dict[str, Any]

    @staticmethod
    def make(layout_info: layout_details, vars: Sequence[str], location: StripPosition) -> strip_label_details:
        variables: dict[str, Any] = {v: str(layout_info.variables[v]) for v in vars}
        meta: dict[str, Any] = {'dimension': 'cols' if location == 'top' else 'rows'}
        return strip_label_details(variables, meta)

    def __len__(self) -> int:
        """
        Number of variables
        """
        return len(self.variables)

    def __copy__(self) -> strip_label_details:
        """
        Make a copy
        """
        return strip_label_details(self.variables.copy(), self.meta.copy())

    def copy(self) -> strip_label_details:
        """
        Make a copy
        """
        return copy(self)

    def text(self) -> str:
        """
        Strip text

        Join the labels for all the variables along a
        dimension
        """
        return '\n'.join(list(self.variables.values()))

    def collapse(self) -> strip_label_details:
        """
        Concatenate all label values into one item
        """
        result = self.copy()
        result.variables = {'value': ', '.join(result.variables.values())}
        return result