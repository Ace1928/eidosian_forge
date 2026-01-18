import os
from . import sysconfig
from .errors import CompileError, DistutilsExecError
from .unixccompiler import UnixCCompiler
distutils.zosccompiler

Contains the selection of the c & c++ compilers on z/OS. There are several
different c compilers on z/OS, all of them are optional, so the correct
one needs to be chosen based on the users input. This is compatible with
the following compilers:

IBM C/C++ For Open Enterprise Languages on z/OS 2.0
IBM Open XL C/C++ 1.1 for z/OS
IBM XL C/C++ V2.4.1 for z/OS 2.4 and 2.5
IBM z/OS XL C/C++
