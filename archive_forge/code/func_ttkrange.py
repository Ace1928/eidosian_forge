import re
import sys
import tkinter
import tkinter.ttk as ttk
from warnings import warn
from .std import TqdmExperimentalWarning, TqdmWarning
from .std import tqdm as std_tqdm
def ttkrange(*args, **kwargs):
    """Shortcut for `tqdm.tk.tqdm(range(*args), **kwargs)`."""
    return tqdm_tk(range(*args), **kwargs)