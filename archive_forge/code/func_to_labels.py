from abc import ABC, abstractmethod
from typing import Callable, Dict, Hashable, List, Optional, Union
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType
@abstractmethod
def to_labels(self, column_labels: Union[str, List[str]]) -> 'ModinDataframe':
    """
        Replace the row labels with one or more columns of data.

        Parameters
        ----------
        column_labels : string or list of strings
            Column label(s) to use as the new row labels.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row labels replaced by the specified columns.

        Notes
        -----
        When multiple column labels are specified, a hierarchical set of labels is created, ordered by the ordering
        of labels in the input.
        """
    pass