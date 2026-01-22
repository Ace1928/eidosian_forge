from typing import Any, Dict, Optional, Sequence

        Get a query that partitions the original `query`.

        Parameters
        ----------
        query : str
            The SQL query to get a partition.
        limit : int
            The size of the partition.
        offset : int
            Where the partition begins.

        Returns
        -------
        str
        