import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
def translate_exprs_to_base(exprs, base):
    """
    Fold expressions.

    Fold expressions with their input nodes until `base`
    frame is the only input frame.

    Parameters
    ----------
    exprs : dict
        Expressions to translate.
    base : HdkOnNativeDataframe
        Required input frame for translated expressions.

    Returns
    -------
    dict
        Translated expressions.
    """
    new_exprs = dict(exprs)
    frames = set()
    for expr in new_exprs.values():
        expr.collect_frames(frames)
    frames.discard(base)
    while len(frames) > 0:
        mapper = InputMapper()
        new_frames = set()
        for frame in frames:
            frame_base = frame._op.input[0]
            if frame_base != base:
                new_frames.add(frame_base)
            assert isinstance(frame._op, TransformNode)
            mapper.add_mapper(frame, TransformMapper(frame._op))
        for k, v in new_exprs.items():
            new_expr = v.translate_input(mapper)
            new_expr.collect_frames(new_frames)
            new_exprs[k] = new_expr
        new_frames.discard(base)
        frames = new_frames
    res = {}
    for col in exprs.keys():
        res[col] = new_exprs[col]
    return res