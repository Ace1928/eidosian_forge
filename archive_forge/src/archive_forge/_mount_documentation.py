from __future__ import absolute_import, division, print_function
import os
Test whether a path is a mount point
    This is a copy of the upstream version of ismount(). Originally this was copied here as a workaround
    until Python issue 2466 was fixed. Now it is here so this will work on older versions of Python
    that may not have the upstream fix.
    https://github.com/ansible/ansible-modules-core/issues/2186
    http://bugs.python.org/issue2466
    