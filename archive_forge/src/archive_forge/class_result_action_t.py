from ctypes import Structure, c_int, c_byte
class result_action_t(Structure):
    _fields_ = [('status', c_int), ('length', c_int), ('value', c_byte * _VALUE_BUFFER_SIZE)]