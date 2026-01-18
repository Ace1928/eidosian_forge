import io
import struct
from qiskit.qpy import formats
def write_generic_typed_data(file_obj, type_key, data_binary):
    """Write statically typed binary data to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        type_key (Enum): Object type of the data.
        data_binary (bytes): Binary data to write.
    """
    data_header = struct.pack(formats.INSTRUCTION_PARAM_PACK, type_key, len(data_binary))
    file_obj.write(data_header)
    file_obj.write(data_binary)