import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _read_records(file: 'pyarrow.NativeFile', path: str) -> Iterable[memoryview]:
    """
    Read records from TFRecord file.

    A TFRecord file contains a sequence of records. The file can only be read
    sequentially. Each record is stored in the following formats:
        uint64 length
        uint32 masked_crc32_of_length
        byte   data[length]
        uint32 masked_crc32_of_data

    See https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details
    for more details.
    """
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)
    row_count = 0
    while True:
        try:
            num_length_bytes_read = file.readinto(length_bytes)
            if num_length_bytes_read == 0:
                break
            elif num_length_bytes_read != 8:
                raise ValueError('Failed to read the length of record data. Expected 8 bytes but got {num_length_bytes_read} bytes.')
            num_length_crc_bytes_read = file.readinto(crc_bytes)
            if num_length_crc_bytes_read != 4:
                raise ValueError('Failed to read the length of CRC-32C hashes. Expected 4 bytes but got {num_length_crc_bytes_read} bytes.')
            data_length, = struct.unpack('<Q', length_bytes)
            if data_length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(data_length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:data_length]
            num_datum_bytes_read = file.readinto(datum_bytes_view)
            if num_datum_bytes_read != data_length:
                raise ValueError(f'Failed to read the record. Exepcted {data_length} bytes but got {num_datum_bytes_read} bytes.')
            num_crc_bytes_read = file.readinto(crc_bytes)
            if num_crc_bytes_read != 4:
                raise ValueError(f'Failed to read the CRC-32C hashes. Expected 4 bytes but got {num_crc_bytes_read} bytes.')
            yield datum_bytes_view
            row_count += 1
            data_length = None
        except Exception as e:
            error_message = f'Failed to read TFRecord file {path}. Please ensure that the TFRecord file has correct format. Already read {row_count} rows.'
            if data_length is not None:
                error_message += f' Byte size of current record data is {data_length}.'
            raise RuntimeError(error_message) from e