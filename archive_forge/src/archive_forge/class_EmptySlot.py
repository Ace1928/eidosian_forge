from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class EmptySlot(FixedSlot):

    def __init__(self, slot_name, py3=True, py2=True, ifdef=None):
        FixedSlot.__init__(self, slot_name, '0', py3=py3, py2=py2, ifdef=ifdef)