from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import warnings
def RunMain():
    _fix_google_module()
    import gslib.__main__
    sys.exit(gslib.__main__.main())