import pandas
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher

        Load an h5 file from the file path or buffer, returning a query compiler.

        Parameters
        ----------
        path_or_buf : str, buffer or path object
            Path to the file to open, or an open :class:`pandas.HDFStore` object.
        **kwargs : dict
            Pass into pandas.read_hdf function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        