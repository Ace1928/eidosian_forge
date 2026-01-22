from typing import Any, Dict, List, Union, Callable
from fugue.dataframe.dataframe import DataFrame
from triad.collections.dict import IndexedOrderedDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
Create another DataFrames with the same structure,
        but all converted by ``func``

        :return: the new DataFrames

        .. admonition:: Examples

            >>> dfs2 = dfs.convert(lambda df: df.as_local()) # convert all to local
        