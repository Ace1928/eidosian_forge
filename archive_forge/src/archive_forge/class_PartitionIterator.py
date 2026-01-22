from __future__ import annotations
from collections.abc import Iterator
from typing import TYPE_CHECKING
class PartitionIterator(Iterator):
    """
    Iterator on partitioned data.

    Parameters
    ----------
    df : modin.pandas.DataFrame
        The dataframe to iterate over.
    axis : {0, 1}
        Axis to iterate over.
    func : callable
        The function to get inner iterables from each partition.
    """
    df: DataFrame

    def __init__(self, df: DataFrame, axis, func):
        self.df = df
        self.axis = axis
        self.index_iter = zip(iter((slice(None) for _ in range(len(self.df.columns)))), range(len(self.df.columns))) if axis else zip(range(len(self.df.index)), iter((slice(None) for _ in range(len(self.df.index)))))
        self.func = func

    def __iter__(self):
        """
        Implement iterator interface.

        Returns
        -------
        PartitionIterator
            Iterator object.
        """
        return self

    def __next__(self):
        """
        Implement iterator interface.

        Returns
        -------
        PartitionIterator
            Incremented iterator object.
        """
        key = next(self.index_iter)
        df = self.df.iloc[key]
        return self.func(df)