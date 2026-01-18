import os
from .base import (
from ..utils.filemanip import fname_presuffix
from ..external.due import BibTeX
Generate a filename based on the given parameters.

        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extensions specified in
        <instance>inputs.output_type.

        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        change_ext : bool
            Flag to change the filename extension to the given `ext`.
            (Default is False)

        Returns
        -------
        fname : str
            New filename based on given parameters.

        