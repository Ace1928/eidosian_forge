import collections
from . import _constants as C
def to_indices(self, tokens):
    """Converts tokens to indices according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """
    to_reduce = False
    if not isinstance(tokens, list):
        tokens = [tokens]
        to_reduce = True
    indices = [self.token_to_idx[token] if token in self.token_to_idx else C.UNKNOWN_IDX for token in tokens]
    return indices[0] if to_reduce else indices