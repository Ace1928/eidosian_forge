import os
from contextlib import AbstractContextManager
from copy import deepcopy
from textwrap import wrap
import re
from datetime import datetime as dt
from dateutil.parser import parse as parseutc
import platform
from ... import logging, config
from ...utils.misc import is_container, rgetcwd
from ...utils.filemanip import md5, hash_infile
def _outputs_help(cls):
    """
    Prints description for output parameters

    >>> from nipype.interfaces.afni import GCOR
    >>> _outputs_help(GCOR)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ['Outputs::', '', '\\tout: (a float)\\n\\t\\tglobal correlation value']

    """
    helpstr = ['Outputs::', '', '\tNone']
    if cls.output_spec:
        outputs = cls.output_spec()
        outhelpstr = ['\n'.join(get_trait_desc(outputs, name, spec)) for name, spec in outputs.traits(transient=None).items()]
        if outhelpstr:
            helpstr = helpstr[:-1] + outhelpstr
    return helpstr