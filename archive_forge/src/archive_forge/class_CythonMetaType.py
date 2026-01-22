from __future__ import absolute_import
import math, sys
class CythonMetaType(type):

    def __getitem__(type, ix):
        return array(type, ix)