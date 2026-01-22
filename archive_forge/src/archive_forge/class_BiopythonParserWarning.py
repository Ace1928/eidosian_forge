import os
import warnings
class BiopythonParserWarning(BiopythonWarning):
    """Biopython parser warning.

    Some in-valid data files cannot be parsed and will trigger an exception.
    Where a reasonable interpretation is possible, Biopython will issue this
    warning to indicate a potential problem. To silence these warnings, use:

    >>> import warnings
    >>> from Bio import BiopythonParserWarning
    >>> warnings.simplefilter('ignore', BiopythonParserWarning)

    Consult the warnings module documentation for more details.
    """