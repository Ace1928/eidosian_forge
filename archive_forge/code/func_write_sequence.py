import io
import struct
from qiskit.qpy import formats
def write_sequence(file_obj, sequence, serializer, **kwargs):
    """Write a sequence of data in the file like object.

    Args:
        file_obj (File): A file like object to write data.
        sequence (Sequence): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
            This must return type key and binary data of each element.
        kwargs: Options set to the serializer.
    """
    num_elements = len(sequence)
    file_obj.write(struct.pack(formats.SEQUENCE_PACK, num_elements))
    for datum in sequence:
        type_key, datum_bytes = serializer(datum, **kwargs)
        write_generic_typed_data(file_obj, type_key, datum_bytes)