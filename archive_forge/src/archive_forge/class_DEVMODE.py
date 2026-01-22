import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class DEVMODE(Structure):
    _anonymous_ = ('_dummyUnion',)
    _fields_ = [('dmDeviceName', BCHAR * CCHDEVICENAME), ('dmSpecVersion', WORD), ('dmDriverVersion', WORD), ('dmSize', WORD), ('dmDriverExtra', WORD), ('dmFields', DWORD), ('_dummyUnion', _DUMMYDEVUNION), ('dmColor', c_short), ('dmDuplex', c_short), ('dmYResolution', c_short), ('dmTTOption', c_short), ('dmCollate', c_short), ('dmFormName', BCHAR * CCHFORMNAME), ('dmLogPixels', WORD), ('dmBitsPerPel', DWORD), ('dmPelsWidth', DWORD), ('dmPelsHeight', DWORD), ('dmDisplayFlags', DWORD), ('dmDisplayFrequency', DWORD), ('dmICMMethod', DWORD), ('dmICMIntent', DWORD), ('dmDitherType', DWORD), ('dmReserved1', DWORD), ('dmReserved2', DWORD), ('dmPanningWidth', DWORD), ('dmPanningHeight', DWORD)]