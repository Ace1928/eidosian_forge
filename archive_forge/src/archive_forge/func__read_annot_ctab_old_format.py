import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def _read_annot_ctab_old_format(fobj, n_entries):
    """Read in an old-style Freesurfer color table from `fobj`.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    This function is used by :func:`read_annot`.

    Parameters
    ----------

    fobj : file-like
        Open file handle to a Freesurfer `.annot` file, with seek point
        at the beginning of the color table data.
    n_entries : int
        Number of entries in the color table.

    Returns
    -------

    ctab : ndarray, shape (n_entries, 5)
        RGBT colortable array - the last column contains all zeros.
    names : list of str
        The names of the labels. The length of the list is n_entries.
    """
    assert hasattr(fobj, 'read')
    dt = _ANNOT_DT
    length = np.fromfile(fobj, dt, 1)[0]
    orig_tab = np.fromfile(fobj, '>c', length)
    orig_tab = orig_tab[:-1]
    names = list()
    ctab = np.zeros((n_entries, 5), dt)
    for i in range(n_entries):
        name_length = np.fromfile(fobj, dt, 1)[0]
        name = np.fromfile(fobj, '|S%d' % name_length, 1)[0]
        names.append(name)
        ctab[i, :4] = np.fromfile(fobj, dt, 4)
    return (ctab, names)