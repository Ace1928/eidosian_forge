import bz2
import json
import lzma
import zlib
from datetime import datetime, timezone
from decimal import Context
from io import BytesIO
from struct import error as StructError
from typing import IO, Union, Optional, Generic, TypeVar, Iterator, Dict
from warnings import warn
from .io.binary_decoder import BinaryDecoder
from .io.json_decoder import AvroJSONDecoder
from .logical_readers import LOGICAL_READERS
from .schema import (
from .types import Schema, AvroMessage, NamedSchemas
from ._read_common import (
from .const import NAMED_TYPES, AVRO_TYPES
class block_reader(file_reader[Block]):
    """Iterator over :class:`.Block` in an avro file.

    Parameters
    ----------
    fo
        Input stream
    reader_schema
        Reader schema
    return_record_name
        If true, when reading a union of records, the result will be a tuple
        where the first value is the name of the record and the second value is
        the record itself
    return_record_name_override
        If true, this will modify the behavior of return_record_name so that
        the record name is only returned for unions where there is more than
        one record. For unions that only have one record, this option will make
        it so that the record is returned by itself, not a tuple with the name.
    return_named_type
        If true, when reading a union of named types, the result will be a tuple
        where the first value is the name of the type and the second value is
        the record itself
        NOTE: Using this option will ignore return_record_name and
        return_record_name_override
    return_named_type_override
        If true, this will modify the behavior of return_named_type so that
        the named type is only returned for unions where there is more than
        one named type. For unions that only have one named type, this option
        will make it so that the named type is returned by itself, not a tuple
        with the name
    handle_unicode_errors
        Default `strict`. Should be set to a valid string that can be used in
        the errors argument of the string decode() function. Examples include
        `replace` and `ignore`


    Example::

        from fastavro import block_reader
        with open('some-file.avro', 'rb') as fo:
            avro_reader = block_reader(fo)
            for block in avro_reader:
                process_block(block)

    .. attribute:: metadata

        Key-value pairs in the header metadata

    .. attribute:: codec

        The codec used when writing

    .. attribute:: writer_schema

        The schema used when writing

    .. attribute:: reader_schema

        The schema used when reading (if provided)
    """

    def __init__(self, fo: IO, reader_schema: Optional[Schema]=None, return_record_name: bool=False, return_record_name_override: bool=False, handle_unicode_errors: str='strict', return_named_type: bool=False, return_named_type_override: bool=False):
        options = {'return_record_name': return_record_name, 'return_record_name_override': return_record_name_override, 'handle_unicode_errors': handle_unicode_errors, 'return_named_type': return_named_type, 'return_named_type_override': return_named_type_override}
        super().__init__(fo, reader_schema, options)
        self._read_header()
        self._elems = _iter_avro_blocks(self.decoder, self._header, self.codec, self.writer_schema, self._named_schemas, self.reader_schema, self.options)