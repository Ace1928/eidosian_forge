from abc import ABC, abstractmethod
from typing import Callable, Dict, Hashable, List, Optional, Union
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType
@abstractmethod
def tree_reduce(self, axis: Union[int, Axis], map_func: Callable, reduce_func: Optional[Callable]=None, dtypes: Optional[str]=None) -> 'ModinDataframe':
    """
        Perform a user-defined aggregation on the specified axis, where the axis reduces down to a singleton using a tree-reduce computation pattern.

        The map function is applied first over multiple partitions of a column, and then the reduce
        function (if specified, otherwise the map function is applied again) is applied to the
        results to produce a single value.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the tree reduce over.
        map_func : callable(row|col) -> row|col|single value
            The map function to apply to each column.
        reduce_func : callable(row|col) -> single value, optional
            The reduce function to apply to the results of the map function.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the same columns as the previous, with only a single row.

        Notes
        -----
        The user-defined function must reduce to a single value.

        If the user-defined function requires access to the entire column, please use reduce instead.
        """
    pass