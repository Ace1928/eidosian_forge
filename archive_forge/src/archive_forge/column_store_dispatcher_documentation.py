import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize

        Build query compiler from deployed tasks outputs.

        Parameters
        ----------
        path : str, path object or file-like object
            Path to the file to read.
        columns : list
            List of columns that should be read from file.
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.
        