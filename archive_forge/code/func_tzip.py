from warnings import warn
from ..auto import tqdm as tqdm_auto
from ..std import TqdmDeprecationWarning, tqdm
from ..utils import ObjectWrapper
def tzip(iter1, *iter2plus, **tqdm_kwargs):
    """
    Equivalent of builtin `zip`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    kwargs = tqdm_kwargs.copy()
    tqdm_class = kwargs.pop('tqdm_class', tqdm_auto)
    for i in zip(tqdm_class(iter1, **kwargs), *iter2plus):
        yield i