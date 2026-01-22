from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def additional_column_validator(df: pd.DataFrame, fields: list[str]=['prompt', 'completion']) -> Remediation:
    """
    This validator will remove additional columns from the dataframe.
    """
    additional_columns = []
    necessary_msg = None
    immediate_msg = None
    necessary_fn = None
    if len(df.columns) > 2:
        additional_columns = [c for c in df.columns if c not in fields]
        warn_message = ''
        for ac in additional_columns:
            dups = [c for c in additional_columns if ac in c]
            if len(dups) > 0:
                warn_message += f'\n  WARNING: Some of the additional columns/keys contain `{ac}` in their name. These will be ignored, and the column/key `{ac}` will be used instead. This could also result from a duplicate column/key in the provided file.'
        immediate_msg = f'\n- The input file should contain exactly two columns/keys per row. Additional columns/keys present are: {additional_columns}{warn_message}'
        necessary_msg = f'Remove additional columns/keys: {additional_columns}'

        def necessary_fn(x: Any) -> Any:
            return x[fields]
    return Remediation(name='additional_column', immediate_msg=immediate_msg, necessary_msg=necessary_msg, necessary_fn=necessary_fn)