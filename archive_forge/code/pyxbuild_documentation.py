import os
import sys
from distutils.errors import DistutilsArgError, DistutilsError, CCompilerError
from distutils.extension import Extension
from distutils.util import grok_environment_error
Compile a PYX file to a DLL and return the name of the generated .so
       or .dll .