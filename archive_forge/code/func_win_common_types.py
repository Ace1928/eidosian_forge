import sys
from . import model
from .error import FFIError
def win_common_types():
    return {'UNICODE_STRING': model.StructType('_UNICODE_STRING', ['Length', 'MaximumLength', 'Buffer'], [model.PrimitiveType('unsigned short'), model.PrimitiveType('unsigned short'), model.PointerType(model.PrimitiveType('wchar_t'))], [-1, -1, -1]), 'PUNICODE_STRING': 'UNICODE_STRING *', 'PCUNICODE_STRING': 'const UNICODE_STRING *', 'TBYTE': 'set-unicode-needed', 'TCHAR': 'set-unicode-needed', 'LPCTSTR': 'set-unicode-needed', 'PCTSTR': 'set-unicode-needed', 'LPTSTR': 'set-unicode-needed', 'PTSTR': 'set-unicode-needed', 'PTBYTE': 'set-unicode-needed', 'PTCHAR': 'set-unicode-needed'}