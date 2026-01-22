from __future__ import print_function, absolute_import, division, unicode_literals
from .compat import no_limit_int  # NOQA
from ruamel.yaml.anchor import Anchor
class DecimalInt(ScalarInt):
    """needed if anchor"""

    def __new__(cls, value, width=None, underscore=None, anchor=None):
        return ScalarInt.__new__(cls, value, width=width, underscore=underscore, anchor=anchor)