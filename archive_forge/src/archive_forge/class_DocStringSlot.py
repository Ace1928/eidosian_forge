from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class DocStringSlot(SlotDescriptor):

    def slot_code(self, scope):
        doc = scope.doc
        if doc is None:
            return '0'
        if doc.is_unicode:
            doc = doc.as_utf8_string()
        return 'PyDoc_STR(%s)' % doc.as_c_string_literal()