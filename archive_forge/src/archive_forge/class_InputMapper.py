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
class InputMapper:
    """
    Input reference mapper.

    This class is used for input translation/replacement in
    expressions via ``BaseExpr.translate_input`` method.

    Translation is performed using column mappers registered via
    `add_mapper` method. Each input frame can have at most one mapper.
    References to frames with no registered mapper are not translated.

    Attributes
    ----------
    _mappers : dict
        Column mappers to use for translation.
    """

    def __init__(self):
        self._mappers = {}

    def add_mapper(self, frame, mapper):
        """
        Register a mapper for a frame.

        Parameters
        ----------
        frame : HdkOnNativeDataframe
            A frame for which a mapper is registered.
        mapper : object
            A mapper to register.
        """
        self._mappers[frame] = mapper

    def translate(self, ref):
        """
        Translate column reference by its name.

        Parameters
        ----------
        ref : InputRefExpr
            A column reference to translate.

        Returns
        -------
        BaseExpr
            Translated expression.
        """
        if ref.modin_frame in self._mappers:
            return self._mappers[ref.modin_frame].translate(ref.column)
        return ref