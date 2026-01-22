import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class ProcThreadAttributeList(object):
    """
    Extended process and thread attribute support.

    To be used with L{STARTUPINFOEX}.
    Only available for Windows Vista and above.

    @type AttributeList: list of tuple( int, ctypes-compatible object )
    @ivar AttributeList: List of (Attribute, Value) pairs.

    @type AttributeListBuffer: L{LPPROC_THREAD_ATTRIBUTE_LIST}
    @ivar AttributeListBuffer: Memory buffer used to store the attribute list.
        L{InitializeProcThreadAttributeList},
        L{UpdateProcThreadAttribute},
        L{DeleteProcThreadAttributeList} and
        L{STARTUPINFOEX}.
    """

    def __init__(self, AttributeList):
        """
        @type  AttributeList: list of tuple( int, ctypes-compatible object )
        @param AttributeList: List of (Attribute, Value) pairs.
        """
        self.AttributeList = AttributeList
        self.AttributeListBuffer = InitializeProcThreadAttributeList(len(AttributeList))
        try:
            for Attribute, Value in AttributeList:
                UpdateProcThreadAttribute(self.AttributeListBuffer, Attribute, Value)
        except:
            ProcThreadAttributeList.__del__(self)
            raise

    def __del__(self):
        try:
            DeleteProcThreadAttributeList(self.AttributeListBuffer)
            del self.AttributeListBuffer
        except Exception:
            pass

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self):
        return self.__class__(self.AttributeList)

    @property
    def value(self):
        return ctypes.cast(ctypes.pointer(self.AttributeListBuffer), LPVOID)

    @property
    def _as_parameter_(self):
        return self.value

    @staticmethod
    def from_param(value):
        raise NotImplementedError()