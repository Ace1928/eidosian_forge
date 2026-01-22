from .. import sanitize
from .. import utils
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
def _matrix_to_data_frame(data, gene_names=None, cell_names=None, sparse=None):
    """Return the optimal data type given data, gene names and cell names.

    Parameters
    ----------
    data : array-like
    gene_names : `str`, array-like or `None` (default: None)
        Either a filename or an array containing a list of gene symbols or ids.
    cell_names : `str`, array-like or `None` (default: None)
        Either a filename or an array containing a list of cell barcodes.
    sparse : `bool` or `None` (default: None)
        If not `None`, overrides default sparsity of the data.
    """
    if gene_names is None and cell_names is None and (not isinstance(data, pd.DataFrame)):
        if sparse is not None:
            if sparse:
                if not sp.issparse(data):
                    data = sp.csr_matrix(data)
            elif sp.issparse(data) and (not sparse):
                data = data.toarray()
        else:
            pass
    else:
        gene_names = _parse_gene_names(gene_names, data)
        cell_names = _parse_cell_names(cell_names, data)
        if sparse is None:
            sparse = utils.is_sparse_dataframe(data) or sp.issparse(data)
        if sparse and gene_names is not None and (len(np.unique(gene_names)) < len(gene_names)):
            warnings.warn('Duplicate gene names detected! Forcing dense matrix.', RuntimeWarning)
            sparse = False
        if cell_names is not None and len(np.unique(cell_names)) < len(cell_names):
            warnings.warn('Duplicate cell names detected! Some functions may not work as intended. You can fix this by running `scprep.sanitize.check_index(data)`.', RuntimeWarning)
        if sparse:
            if isinstance(data, pd.DataFrame):
                if gene_names is not None:
                    data.columns = gene_names
                if cell_names is not None:
                    data.index = cell_names
                if not utils.is_sparse_dataframe(data):
                    data = utils.dataframe_to_sparse(data, fill_value=0.0)
            elif sp.issparse(data):
                data = pd.DataFrame.sparse.from_spmatrix(data, index=cell_names, columns=gene_names)
            else:
                data = pd.DataFrame(data, index=cell_names, columns=gene_names)
                data = utils.dataframe_to_sparse(data, fill_value=0.0)
        elif isinstance(data, pd.DataFrame):
            if gene_names is not None:
                data.columns = gene_names
            if cell_names is not None:
                data.index = cell_names
            if utils.is_sparse_dataframe(data):
                data = data.sparse.to_dense()
        else:
            if sp.issparse(data):
                data = data.toarray()
            data = pd.DataFrame(data, index=cell_names, columns=gene_names)
    data = sanitize.check_numeric(data, suppress_errors=True)
    return data