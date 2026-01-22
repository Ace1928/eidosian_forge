from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def WriteVtable(self):
    """
        WriteVtable serializes the vtable for the current object, if needed.

        Before writing out the vtable, this checks pre-existing vtables for
        equality to this one. If an equal vtable is found, point the object to
        the existing vtable and return.

        Because vtable values are sensitive to alignment of object data, not
        all logically-equal vtables will be deduplicated.

        A vtable has the following format:
          <VOffsetT: size of the vtable in bytes, including this value>
          <VOffsetT: size of the object in bytes, including the vtable offset>
          <VOffsetT: offset for a field> * N, where N is the number of fields
                     in the schema for this type. Includes deprecated fields.
        Thus, a vtable is made of 2 + N elements, each VOffsetT bytes wide.

        An object has the following format:
          <SOffsetT: offset to this object's vtable (may be negative)>
          <byte: data>+
        """
    self.PrependSOffsetTRelative(0)
    objectOffset = self.Offset()
    vtKey = []
    trim = True
    for elem in reversed(self.current_vtable):
        if elem == 0:
            if trim:
                continue
        else:
            elem = objectOffset - elem
            trim = False
        vtKey.append(elem)
    vtKey = tuple(vtKey)
    vt2Offset = self.vtables.get(vtKey)
    if vt2Offset is None:
        i = len(self.current_vtable) - 1
        trailing = 0
        trim = True
        while i >= 0:
            off = 0
            elem = self.current_vtable[i]
            i -= 1
            if elem == 0:
                if trim:
                    trailing += 1
                    continue
            else:
                off = objectOffset - elem
                trim = False
            self.PrependVOffsetT(off)
        objectSize = UOffsetTFlags.py_type(objectOffset - self.objectEnd)
        self.PrependVOffsetT(VOffsetTFlags.py_type(objectSize))
        vBytes = len(self.current_vtable) - trailing + VtableMetadataFields
        vBytes *= N.VOffsetTFlags.bytewidth
        self.PrependVOffsetT(VOffsetTFlags.py_type(vBytes))
        objectStart = SOffsetTFlags.py_type(len(self.Bytes) - objectOffset)
        encode.Write(packer.soffset, self.Bytes, objectStart, SOffsetTFlags.py_type(self.Offset() - objectOffset))
        self.vtables[vtKey] = self.Offset()
    else:
        objectStart = SOffsetTFlags.py_type(len(self.Bytes) - objectOffset)
        self.head = UOffsetTFlags.py_type(objectStart)
        encode.Write(packer.soffset, self.Bytes, self.Head(), SOffsetTFlags.py_type(vt2Offset - objectOffset))
    self.current_vtable = None
    return objectOffset