from .. import utils
from .utils import _matrix_to_data_frame
import pandas as pd
Load a tsv file.

    Parameters
    ----------
    filename : str
        The name of the csv file to be loaded
    cell_axis : {'row', 'column'}, optional (default: 'row')
        If your data has genes on the rows and cells on the columns, use
        cell_axis='column'
    delimiter : str, optional (default: '\t')
        Use ',' for comma separated values (csv)
    gene_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume gene names are in the first row/column. Otherwise
        expects a filename or an array containing a list of gene symbols or ids
    cell_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume cell names are in the first row/column. Otherwise
        expects a filename or an array containing a list of cell barcodes.
    sparse : bool, optional (default: False)
        If True, loads the data as a pd.DataFrame[pd.SparseArray]. This uses less memory
        but more CPU.
    **kwargs : optional arguments for `pd.read_csv`.

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        If either gene or cell names are given, data will be a pd.DataFrame or
        pd.DataFrame[pd.SparseArray]. If no names are given, data will be a np.ndarray
        or scipy.sparse.spmatrix
    