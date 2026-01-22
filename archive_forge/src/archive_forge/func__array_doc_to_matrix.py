from ._sputils import isintlike, isscalarlike
def _array_doc_to_matrix(docstr):
    if docstr is None:
        return None
    return docstr.replace('sparse arrays', 'sparse matrices').replace('sparse array', 'sparse matrix')