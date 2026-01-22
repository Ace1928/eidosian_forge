from __future__ import absolute_import
import re
from decimal import Decimal, getcontext
from functools import partial
class _EOF(object):

    def __repr__(self):
        return 'EOF'