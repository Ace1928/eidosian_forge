from __future__ import annotations
import keyword
import warnings
from typing import Collection, List, Mapping, Optional, Set, Tuple, Union
def add_axis_name(x: str) -> None:
    if x in self.identifiers:
        if not (allow_underscore and x == '_') and (not allow_duplicates):
            raise ValueError(f"Indexing expression contains duplicate dimension '{x}'")
    if x == _ellipsis:
        self.identifiers.add(_ellipsis)
        if bracket_group is None:
            self.composition.append(_ellipsis)
            self.has_ellipsis_parenthesized = False
        else:
            bracket_group.append(_ellipsis)
            self.has_ellipsis_parenthesized = True
    else:
        is_number = str.isdecimal(x)
        if is_number and int(x) == 1:
            if bracket_group is None:
                self.composition.append([])
            else:
                pass
            return
        is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
        if not (is_number or is_axis_name):
            raise ValueError(f'Invalid axis identifier: {x}\n{reason}')
        axis_name: Union[str, AnonymousAxis] = AnonymousAxis(x) if is_number else x
        self.identifiers.add(axis_name)
        if is_number:
            self.has_non_unitary_anonymous_axes = True
        if bracket_group is None:
            self.composition.append([axis_name])
        else:
            bracket_group.append(axis_name)