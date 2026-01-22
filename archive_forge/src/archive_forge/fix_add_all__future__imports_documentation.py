from lib2to3 import fixer_base
from libfuturize.fixer_util import future_import

Fixer for adding:

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    from __future__ import unicode_literals

This is done when converting from Py3 to both Py3/Py2.
