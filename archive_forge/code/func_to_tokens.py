import collections
from . import _constants as C
def to_tokens(self, indices):
    """Converts token indices to tokens according to the vocabulary.


        Parameters
        ----------
        indices : int or list of ints
            A source token index or token indices to be converted.


        Returns
        -------
        str or list of strs
            A token or a list of tokens according to the vocabulary.
        """
    to_reduce = False
    if not isinstance(indices, list):
        indices = [indices]
        to_reduce = True
    max_idx = len(self.idx_to_token) - 1
    tokens = []
    for idx in indices:
        if not isinstance(idx, int) or idx > max_idx:
            raise ValueError('Token index %d in the provided `indices` is invalid.' % idx)
        tokens.append(self.idx_to_token[idx])
    return tokens[0] if to_reduce else tokens