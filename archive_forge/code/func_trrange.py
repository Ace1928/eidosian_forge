from warnings import warn
from rich.progress import (
from .std import TqdmExperimentalWarning
from .std import tqdm as std_tqdm
def trrange(*args, **kwargs):
    """Shortcut for `tqdm.rich.tqdm(range(*args), **kwargs)`."""
    return tqdm_rich(range(*args), **kwargs)