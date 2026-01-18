import json
import struct
import zlib
import warnings
from io import BytesIO
import numpy as np
import symengine as sym
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.exceptions import QiskitError
from qiskit.pulse import library, channels, instructions
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.qpy.exceptions import QpyError
from qiskit.pulse.configuration import Kernel, Discriminator
def write_schedule_block(file_obj, block, metadata_serializer=None, use_symengine=False, version=common.QPY_VERSION):
    """Write a single ScheduleBlock object in the file like object.

    Args:
        file_obj (File): The file like object to write the circuit data in.
        block (ScheduleBlock): A schedule block data to write.
        metadata_serializer (JSONEncoder): An optional JSONEncoder class that
            will be passed the :attr:`.ScheduleBlock.metadata` dictionary for
            ``block`` and will be used as the ``cls`` kwarg
            on the ``json.dump()`` call to JSON serialize that dictionary.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        version (int): The QPY format version to use for serializing this circuit block
    Raises:
        TypeError: If any of the instructions is invalid data format.
    """
    metadata = json.dumps(block.metadata, separators=(',', ':'), cls=metadata_serializer).encode(common.ENCODE)
    block_name = block.name.encode(common.ENCODE)
    header_raw = formats.SCHEDULE_BLOCK_HEADER(name_size=len(block_name), metadata_size=len(metadata), num_elements=len(block))
    header = struct.pack(formats.SCHEDULE_BLOCK_HEADER_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(block_name)
    file_obj.write(metadata)
    _write_alignment_context(file_obj, block.alignment_context)
    for block_elm in block._blocks:
        _write_element(file_obj, block_elm, metadata_serializer, use_symengine)
    flat_key_refdict = {}
    for ref_keys, schedule in block._reference_manager.items():
        key_str = instructions.Reference.key_delimiter.join(ref_keys)
        flat_key_refdict[key_str] = schedule
    common.write_mapping(file_obj=file_obj, mapping=flat_key_refdict, serializer=_dumps_reference_item, metadata_serializer=metadata_serializer)