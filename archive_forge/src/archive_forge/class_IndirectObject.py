import binascii
import codecs
import hashlib
import re
from binascii import unhexlify
from math import log10
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union, cast
from .._codecs import _pdfdoc_encoding_rev
from .._protocols import PdfObjectProtocol, PdfWriterProtocol
from .._utils import (
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
class IndirectObject(PdfObject):

    def __init__(self, idnum: int, generation: int, pdf: Any) -> None:
        self.idnum = idnum
        self.generation = generation
        self.pdf = pdf

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'IndirectObject':
        """Clone object into pdf_dest."""
        if self.pdf == pdf_dest and (not force_duplicate):
            return self
        if id(self.pdf) not in pdf_dest._id_translated:
            pdf_dest._id_translated[id(self.pdf)] = {}
        if self.idnum in pdf_dest._id_translated[id(self.pdf)]:
            dup = pdf_dest.get_object(pdf_dest._id_translated[id(self.pdf)][self.idnum])
            if force_duplicate:
                assert dup is not None
                assert dup.indirect_reference is not None
                idref = dup.indirect_reference
                return IndirectObject(idref.idnum, idref.generation, idref.pdf)
        else:
            obj = self.get_object()
            if obj is None:
                obj = NullObject()
                assert isinstance(self, (IndirectObject,))
                obj.indirect_reference = self
            dup = pdf_dest._add_object(obj.clone(pdf_dest, force_duplicate, ignore_fields))
        assert dup is not None
        assert dup.indirect_reference is not None
        return dup.indirect_reference

    @property
    def indirect_reference(self) -> 'IndirectObject':
        return self

    def get_object(self) -> Optional['PdfObject']:
        return self.pdf.get_object(self)

    def __deepcopy__(self, memo: Any) -> 'IndirectObject':
        return IndirectObject(self.idnum, self.generation, self.pdf)

    def _get_object_with_check(self) -> Optional['PdfObject']:
        o = self.get_object()
        if isinstance(o, IndirectObject):
            raise PdfStreamError(f'{self.__repr__()} references an IndirectObject {o.__repr__()}')
        return o

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._get_object_with_check(), name)
        except AttributeError:
            raise AttributeError(f'No attribute {name} found in IndirectObject or pointed object')

    def __getitem__(self, key: Any) -> Any:
        return self._get_object_with_check()[key]

    def __str__(self) -> str:
        return self.get_object().__str__()

    def __repr__(self) -> str:
        return f'IndirectObject({self.idnum!r}, {self.generation!r}, {id(self.pdf)})'

    def __eq__(self, other: object) -> bool:
        return other is not None and isinstance(other, IndirectObject) and (self.idnum == other.idnum) and (self.generation == other.generation) and (self.pdf is other.pdf)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(f'{self.idnum} {self.generation} R'.encode())

    @staticmethod
    def read_from_stream(stream: StreamType, pdf: Any) -> 'IndirectObject':
        idnum = b''
        while True:
            tok = stream.read(1)
            if not tok:
                raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
            if tok.isspace():
                break
            idnum += tok
        generation = b''
        while True:
            tok = stream.read(1)
            if not tok:
                raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
            if tok.isspace():
                if not generation:
                    continue
                break
            generation += tok
        r = read_non_whitespace(stream)
        if r != b'R':
            raise PdfReadError(f'Error reading indirect object reference at byte {hex(stream.tell())}')
        return IndirectObject(int(idnum), int(generation), pdf)