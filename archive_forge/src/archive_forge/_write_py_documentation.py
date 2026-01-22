from abc import ABC, abstractmethod
import json
from io import BytesIO
from os import urandom, SEEK_SET
import bz2
import lzma
import zlib
from typing import Union, IO, Iterable, Any, Optional, Dict
from warnings import warn
from .const import NAMED_TYPES
from .io.binary_encoder import BinaryEncoder
from .io.json_encoder import AvroJSONEncoder
from .validation import _validate
from .read import HEADER_SCHEMA, SYNC_SIZE, MAGIC, reader
from .logical_writers import LOGICAL_WRITERS
from .schema import extract_record_type, extract_logical_type, parse_schema
from ._write_common import _is_appendable
from .types import Schema, NamedSchemas
Write a single record without the schema or header information

    Parameters
    ----------
    fo
        Output file
    schema
        Schema
    record
        Record to write
    strict
        If set to True, an error will be raised if records do not contain
        exactly the same fields that the schema states
    strict_allow_default
        If set to True, an error will be raised if records do not contain
        exactly the same fields that the schema states unless it is a missing
        field that has a default value in the schema
    disable_tuple_notation
        If set to True, tuples will not be treated as a special case. Therefore,
        using a tuple to indicate the type of a record will not work


    Example::

        parsed_schema = fastavro.parse_schema(schema)
        with open('file', 'wb') as fp:
            fastavro.schemaless_writer(fp, parsed_schema, record)

    Note: The ``schemaless_writer`` can only write a single record.
    