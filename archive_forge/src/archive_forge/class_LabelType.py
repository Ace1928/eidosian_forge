import struct
from llvmlite.ir._utils import _StrCaching
class LabelType(Type):
    """
    The label type is the type of e.g. basic blocks.
    """

    def _to_string(self):
        return 'label'