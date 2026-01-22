import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore
@magic_arguments()
@argument('outputs', nargs='*')
@line_magic
def Rpull(self, line):
    """
        A line-level magic for R that pulls
        variables from python to rpy2::

            In [18]: _ = %R x = c(3,4,6.7); y = c(4,6,7); z = c('a',3,4)

            In [19]: %Rpull x  y z

            In [20]: x
            Out[20]: array([ 3. ,  4. ,  6.7])

            In [21]: y
            Out[21]: array([ 4.,  6.,  7.])

            In [22]: z
            Out[22]:
            array(['a', '3', '4'],
                  dtype='|S1')


        This is useful when a structured array is desired as output, or
        when the object in R has mixed data types.
        See the %%R docstring for more examples.

        Notes
        -----

        Beware that R names can have dots ('.') so this is not fool proof.
        To avoid this, don't name your R objects with dots...

        """
    args = parse_argstring(self.Rpull, line)
    outputs = args.outputs
    with localconverter(self.converter):
        for output in outputs:
            robj = ri.globalenv.find(output)
            self.shell.push({output: robj})