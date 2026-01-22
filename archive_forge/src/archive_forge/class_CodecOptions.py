from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
class CodecOptions(_BaseCodecOptions):
    """Encapsulates options used encoding and / or decoding BSON."""

    def __init__(self, *args, **kwargs):
        """Encapsulates options used encoding and / or decoding BSON.

            The `document_class` option is used to define a custom type for use
            decoding BSON documents. Access to the underlying raw BSON bytes for
            a document is available using the :class:`~bson.raw_bson.RawBSONDocument`
            type::

              >>> from bson.raw_bson import RawBSONDocument
              >>> from bson.codec_options import CodecOptions
              >>> codec_options = CodecOptions(document_class=RawBSONDocument)
              >>> coll = db.get_collection('test', codec_options=codec_options)
              >>> doc = coll.find_one()
              >>> doc.raw
              '\\x16\\x00\\x00\\x00\\x07_id\\x00[0\\x165\\x91\\x10\\xea\\x14\\xe8\\xc5\\x8b\\x93\\x00'

            The document class can be any type that inherits from
            :class:`~collections.abc.MutableMapping`::

              >>> class AttributeDict(dict):
              ...     # A dict that supports attribute access.
              ...     def __getattr__(self, key):
              ...         return self[key]
              ...     def __setattr__(self, key, value):
              ...         self[key] = value
              ...
              >>> codec_options = CodecOptions(document_class=AttributeDict)
              >>> coll = db.get_collection('test', codec_options=codec_options)
              >>> doc = coll.find_one()
              >>> doc._id
              ObjectId('5b3016359110ea14e8c58b93')

            See :doc:`/examples/datetimes` for examples using the `tz_aware` and
            `tzinfo` options.

            See :doc:`/examples/uuid` for examples using the `uuid_representation`
            option.

            :Parameters:
              - `document_class`: BSON documents returned in queries will be decoded
                to an instance of this class. Must be a subclass of
                :class:`~collections.abc.MutableMapping`. Defaults to :class:`dict`.
              - `tz_aware`: If ``True``, BSON datetimes will be decoded to timezone
                aware instances of :class:`~datetime.datetime`. Otherwise they will be
                naive. Defaults to ``False``.
              - `uuid_representation`: The BSON representation to use when encoding
                and decoding instances of :class:`~uuid.UUID`. Defaults to
                :data:`~bson.binary.UuidRepresentation.UNSPECIFIED`. New
                applications should consider setting this to
                :data:`~bson.binary.UuidRepresentation.STANDARD` for cross language
                compatibility. See :ref:`handling-uuid-data-example` for details.
              - `unicode_decode_error_handler`: The error handler to apply when
                a Unicode-related error occurs during BSON decoding that would
                otherwise raise :exc:`UnicodeDecodeError`. Valid options include
                'strict', 'replace', 'backslashreplace', 'surrogateescape', and
                'ignore'. Defaults to 'strict'.
              - `tzinfo`: A :class:`~datetime.tzinfo` subclass that specifies the
                timezone to/from which :class:`~datetime.datetime` objects should be
                encoded/decoded.
              - `type_registry`: Instance of :class:`TypeRegistry` used to customize
                encoding and decoding behavior.
              - `datetime_conversion`: Specifies how UTC datetimes should be decoded
                within BSON. Valid options include 'datetime_ms' to return as a
                DatetimeMS, 'datetime' to return as a datetime.datetime and
                raising a ValueError for out-of-range values, 'datetime_auto' to
                return DatetimeMS objects when the underlying datetime is
                out-of-range and 'datetime_clamp' to clamp to the minimum and
                maximum possible datetimes. Defaults to 'datetime'.

            .. versionchanged:: 4.0
               The default for `uuid_representation` was changed from
               :const:`~bson.binary.UuidRepresentation.PYTHON_LEGACY` to
               :const:`~bson.binary.UuidRepresentation.UNSPECIFIED`.

            .. versionadded:: 3.8
               `type_registry` attribute.

            .. warning:: Care must be taken when changing
               `unicode_decode_error_handler` from its default value ('strict').
               The 'replace' and 'ignore' modes should not be used when documents
               retrieved from the server will be modified in the client application
               and stored back to the server.
            """
        super().__init__()

    def __new__(cls: Type[CodecOptions], document_class: Optional[Type[Mapping[str, Any]]]=None, tz_aware: bool=False, uuid_representation: Optional[int]=UuidRepresentation.UNSPECIFIED, unicode_decode_error_handler: str='strict', tzinfo: Optional[datetime.tzinfo]=None, type_registry: Optional[TypeRegistry]=None, datetime_conversion: Optional[DatetimeConversion]=DatetimeConversion.DATETIME) -> CodecOptions:
        doc_class = document_class or dict
        is_mapping = False
        try:
            is_mapping = issubclass(doc_class, _MutableMapping)
        except TypeError:
            if hasattr(doc_class, '__origin__'):
                is_mapping = issubclass(doc_class.__origin__, _MutableMapping)
        if not (is_mapping or _raw_document_class(doc_class)):
            raise TypeError('document_class must be dict, bson.son.SON, bson.raw_bson.RawBSONDocument, or a subclass of collections.abc.MutableMapping')
        if not isinstance(tz_aware, bool):
            raise TypeError(f'tz_aware must be True or False, was: tz_aware={tz_aware}')
        if uuid_representation not in ALL_UUID_REPRESENTATIONS:
            raise ValueError('uuid_representation must be a value from bson.binary.UuidRepresentation')
        if not isinstance(unicode_decode_error_handler, str):
            raise ValueError('unicode_decode_error_handler must be a string')
        if tzinfo is not None:
            if not isinstance(tzinfo, datetime.tzinfo):
                raise TypeError('tzinfo must be an instance of datetime.tzinfo')
            if not tz_aware:
                raise ValueError('cannot specify tzinfo without also setting tz_aware=True')
        type_registry = type_registry or TypeRegistry()
        if not isinstance(type_registry, TypeRegistry):
            raise TypeError('type_registry must be an instance of TypeRegistry')
        return tuple.__new__(cls, (doc_class, tz_aware, uuid_representation, unicode_decode_error_handler, tzinfo, type_registry, datetime_conversion))

    def _arguments_repr(self) -> str:
        """Representation of the arguments used to create this object."""
        document_class_repr = 'dict' if self.document_class is dict else repr(self.document_class)
        uuid_rep_repr = UUID_REPRESENTATION_NAMES.get(self.uuid_representation, self.uuid_representation)
        return 'document_class={}, tz_aware={!r}, uuid_representation={}, unicode_decode_error_handler={!r}, tzinfo={!r}, type_registry={!r}, datetime_conversion={!s}'.format(document_class_repr, self.tz_aware, uuid_rep_repr, self.unicode_decode_error_handler, self.tzinfo, self.type_registry, self.datetime_conversion)

    def _options_dict(self) -> dict[str, Any]:
        """Dictionary of the arguments used to create this object."""
        return {'document_class': self.document_class, 'tz_aware': self.tz_aware, 'uuid_representation': self.uuid_representation, 'unicode_decode_error_handler': self.unicode_decode_error_handler, 'tzinfo': self.tzinfo, 'type_registry': self.type_registry, 'datetime_conversion': self.datetime_conversion}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._arguments_repr()})'

    def with_options(self, **kwargs: Any) -> CodecOptions:
        """Make a copy of this CodecOptions, overriding some options::

                >>> from bson.codec_options import DEFAULT_CODEC_OPTIONS
                >>> DEFAULT_CODEC_OPTIONS.tz_aware
                False
                >>> options = DEFAULT_CODEC_OPTIONS.with_options(tz_aware=True)
                >>> options.tz_aware
                True

            .. versionadded:: 3.5
            """
        opts = self._options_dict()
        opts.update(kwargs)
        return CodecOptions(**opts)