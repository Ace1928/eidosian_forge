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
class UnionNode(DFAlgNode):
    """
    A node to represent rows union of input frames.

    Parameters
    ----------
    frames : list of HdkOnNativeDataframe
        Input frames.
    columns : dict
        Column names and dtypes.
    ignore_index : bool

    Attributes
    ----------
    input : list of HdkOnNativeDataframe
        Input frames.
    """

    def __init__(self, frames: List['HdkOnNativeDataframe'], columns: Dict[str, np.dtype], ignore_index: bool):
        self.input = frames
        self.columns = columns
        self.ignore_index = ignore_index

    @_inherit_docstrings(DFAlgNode.require_executed_base)
    def require_executed_base(self) -> bool:
        return not self.can_execute_hdk()

    @_inherit_docstrings(DFAlgNode.can_execute_hdk)
    def can_execute_hdk(self) -> bool:
        if len(self.input) > 2:
            return False
        if len(self.input) == 0 or len(self.columns) == 0:
            return False
        dtypes = self.input[0]._dtypes.to_dict()
        if any((is_string_dtype(t) for t in dtypes.values())) or any((f._dtypes.to_dict() != dtypes for f in self.input[1:])):
            return False
        return True

    @_inherit_docstrings(DFAlgNode.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return True

    def execute_arrow(self, tables: Union[pa.Table, List[pa.Table]]) -> pa.Table:
        """
        Concat frames' rows using Arrow API.

        Parameters
        ----------
        tables : pa.Table or list of pa.Table

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        if len(self.columns) == 0:
            frames = self.input
            if len(frames) == 0:
                return EMPTY_ARROW_TABLE
            elif self.ignore_index:
                idx = pandas.RangeIndex(0, sum((len(frame.index) for frame in frames)))
            else:
                idx = frames[0].index.append([f.index for f in frames[1:]])
            idx_cols = ColNameCodec.mangle_index_names(idx.names)
            idx_df = pandas.DataFrame(index=idx).reset_index()
            obj_cols = idx_df.select_dtypes(include=['object']).columns.tolist()
            if len(obj_cols) != 0:
                idx_df[obj_cols] = idx_df[obj_cols].astype(str)
            idx_table = pa.Table.from_pandas(idx_df, preserve_index=False)
            return idx_table.rename_columns(idx_cols)
        if isinstance(tables, pa.Table):
            assert len(self.input) == 1
            return tables
        try:
            return pa.concat_tables(tables)
        except pa.lib.ArrowInvalid:
            fields: Dict[str, pa.Field] = {}
            for table in tables:
                for col_name in table.column_names:
                    field = table.field(col_name)
                    cur_field = fields.get(col_name, None)
                    if cur_field is None or cur_field.type != get_common_arrow_type(cur_field.type, field.type):
                        fields[col_name] = field
            schema = pa.schema(list(fields.values()))
            for i, table in enumerate(tables):
                tables[i] = pa.table(table.columns, schema=schema)
            return pa.concat_tables(tables)

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        UnionNode
        """
        return UnionNode(self.input, self.columns, self.ignore_index)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return f'{prefix}UnionNode:\n' + self._prints_input(prefix + '  ')