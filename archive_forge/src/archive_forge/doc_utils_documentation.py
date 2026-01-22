from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string

    Build decorator which adds docstring for the groupby reduce method.

    Parameters
    ----------
    result : str
        The result of reduce.
    refer_to : str
        Method name in ``modin.pandas.groupby`` module to refer to
        for more information about parameters and output format.
    action : str, optional
        What method does with groups.

    Returns
    -------
    callable
    