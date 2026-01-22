import ast
import re
from collections import OrderedDict
Parse the 'ASCCONV' format from `input_str`.

    Parameters
    ----------
    ascconv_str : str
        The string we are parsing
    str_delim : str, optional
        String delimiter.  Typically '"' or '""'

    Returns
    -------
    prot_dict : OrderedDict
        Meta data pulled from the ASCCONV section.
    attrs : OrderedDict
        Any attributes stored in the 'ASCCONV BEGIN' line

    Raises
    ------
    AsconvParseError
        A line of the ASCCONV section could not be parsed.
    