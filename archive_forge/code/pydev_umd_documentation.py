import sys
import os
from _pydev_bundle._pydev_execfile import execfile

        Del user modules to force Python to deeply reload them

        Do not del modules which are considered as system modules, i.e.
        modules installed in subdirectories of Python interpreter's binary
        Do not del C modules
        