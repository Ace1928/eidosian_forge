from __future__ import annotations
import base64
import datetime
import json
import math
import re
import uuid
from typing import (
from bson.binary import ALL_UUID_SUBTYPES, UUID_SUBTYPE, Binary, UuidRepresentation
from bson.code import Code
from bson.codec_options import CodecOptions, DatetimeConversion
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
class JSONOptions(_BASE_CLASS):
    json_mode: int
    strict_number_long: bool
    datetime_representation: int
    strict_uuid: bool
    document_class: Type[MutableMapping[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any):
        """Encapsulates JSON options for :func:`dumps` and :func:`loads`.

        :Parameters:
          - `strict_number_long`: If ``True``, :class:`~bson.int64.Int64` objects
            are encoded to MongoDB Extended JSON's *Strict mode* type
            `NumberLong`, ie ``'{"$numberLong": "<number>" }'``. Otherwise they
            will be encoded as an `int`. Defaults to ``False``.
          - `datetime_representation`: The representation to use when encoding
            instances of :class:`datetime.datetime`. Defaults to
            :const:`~DatetimeRepresentation.LEGACY`.
          - `strict_uuid`: If ``True``, :class:`uuid.UUID` object are encoded to
            MongoDB Extended JSON's *Strict mode* type `Binary`. Otherwise it
            will be encoded as ``'{"$uuid": "<hex>" }'``. Defaults to ``False``.
          - `json_mode`: The :class:`JSONMode` to use when encoding BSON types to
            Extended JSON. Defaults to :const:`~JSONMode.LEGACY`.
          - `document_class`: BSON documents returned by :func:`loads` will be
            decoded to an instance of this class. Must be a subclass of
            :class:`collections.MutableMapping`. Defaults to :class:`dict`.
          - `uuid_representation`: The :class:`~bson.binary.UuidRepresentation`
            to use when encoding and decoding instances of :class:`uuid.UUID`.
            Defaults to :const:`~bson.binary.UuidRepresentation.UNSPECIFIED`.
          - `tz_aware`: If ``True``, MongoDB Extended JSON's *Strict mode* type
            `Date` will be decoded to timezone aware instances of
            :class:`datetime.datetime`. Otherwise they will be naive. Defaults
            to ``False``.
          - `tzinfo`: A :class:`datetime.tzinfo` subclass that specifies the
            timezone from which :class:`~datetime.datetime` objects should be
            decoded. Defaults to :const:`~bson.tz_util.utc`.
          - `datetime_conversion`: Specifies how UTC datetimes should be decoded
            within BSON. Valid options include 'datetime_ms' to return as a
            DatetimeMS, 'datetime' to return as a datetime.datetime and
            raising a ValueError for out-of-range values, 'datetime_auto' to
            return DatetimeMS objects when the underlying datetime is
            out-of-range and 'datetime_clamp' to clamp to the minimum and
            maximum possible datetimes. Defaults to 'datetime'. See
            :ref:`handling-out-of-range-datetimes` for details.
          - `args`: arguments to :class:`~bson.codec_options.CodecOptions`
          - `kwargs`: arguments to :class:`~bson.codec_options.CodecOptions`

        .. seealso:: The specification for Relaxed and Canonical `Extended JSON`_.

        .. versionchanged:: 4.0
           The default for `json_mode` was changed from :const:`JSONMode.LEGACY`
           to :const:`JSONMode.RELAXED`.
           The default for `uuid_representation` was changed from
           :const:`~bson.binary.UuidRepresentation.PYTHON_LEGACY` to
           :const:`~bson.binary.UuidRepresentation.UNSPECIFIED`.

        .. versionchanged:: 3.5
           Accepts the optional parameter `json_mode`.

        .. versionchanged:: 4.0
           Changed default value of `tz_aware` to False.
        """
        super().__init__()

    def __new__(cls: Type[JSONOptions], strict_number_long: Optional[bool]=None, datetime_representation: Optional[int]=None, strict_uuid: Optional[bool]=None, json_mode: int=JSONMode.RELAXED, *args: Any, **kwargs: Any) -> JSONOptions:
        kwargs['tz_aware'] = kwargs.get('tz_aware', False)
        if kwargs['tz_aware']:
            kwargs['tzinfo'] = kwargs.get('tzinfo', utc)
        if datetime_representation not in (DatetimeRepresentation.LEGACY, DatetimeRepresentation.NUMBERLONG, DatetimeRepresentation.ISO8601, None):
            raise ValueError('JSONOptions.datetime_representation must be one of LEGACY, NUMBERLONG, or ISO8601 from DatetimeRepresentation.')
        self = cast(JSONOptions, super().__new__(cls, *args, **kwargs))
        if json_mode not in (JSONMode.LEGACY, JSONMode.RELAXED, JSONMode.CANONICAL):
            raise ValueError('JSONOptions.json_mode must be one of LEGACY, RELAXED, or CANONICAL from JSONMode.')
        self.json_mode = json_mode
        if self.json_mode == JSONMode.RELAXED:
            if strict_number_long:
                raise ValueError('Cannot specify strict_number_long=True with JSONMode.RELAXED')
            if datetime_representation not in (None, DatetimeRepresentation.ISO8601):
                raise ValueError('datetime_representation must be DatetimeRepresentation.ISO8601 or omitted with JSONMode.RELAXED')
            if strict_uuid not in (None, True):
                raise ValueError('Cannot specify strict_uuid=False with JSONMode.RELAXED')
            self.strict_number_long = False
            self.datetime_representation = DatetimeRepresentation.ISO8601
            self.strict_uuid = True
        elif self.json_mode == JSONMode.CANONICAL:
            if strict_number_long not in (None, True):
                raise ValueError('Cannot specify strict_number_long=False with JSONMode.RELAXED')
            if datetime_representation not in (None, DatetimeRepresentation.NUMBERLONG):
                raise ValueError('datetime_representation must be DatetimeRepresentation.NUMBERLONG or omitted with JSONMode.RELAXED')
            if strict_uuid not in (None, True):
                raise ValueError('Cannot specify strict_uuid=False with JSONMode.RELAXED')
            self.strict_number_long = True
            self.datetime_representation = DatetimeRepresentation.NUMBERLONG
            self.strict_uuid = True
        else:
            self.strict_number_long = False
            self.datetime_representation = DatetimeRepresentation.LEGACY
            self.strict_uuid = False
            if strict_number_long is not None:
                self.strict_number_long = strict_number_long
            if datetime_representation is not None:
                self.datetime_representation = datetime_representation
            if strict_uuid is not None:
                self.strict_uuid = strict_uuid
        return self

    def _arguments_repr(self) -> str:
        return 'strict_number_long={!r}, datetime_representation={!r}, strict_uuid={!r}, json_mode={!r}, {}'.format(self.strict_number_long, self.datetime_representation, self.strict_uuid, self.json_mode, super()._arguments_repr())

    def _options_dict(self) -> dict[Any, Any]:
        options_dict = super()._options_dict()
        options_dict.update({'strict_number_long': self.strict_number_long, 'datetime_representation': self.datetime_representation, 'strict_uuid': self.strict_uuid, 'json_mode': self.json_mode})
        return options_dict

    def with_options(self, **kwargs: Any) -> JSONOptions:
        """
        Make a copy of this JSONOptions, overriding some options::

            >>> from bson.json_util import CANONICAL_JSON_OPTIONS
            >>> CANONICAL_JSON_OPTIONS.tz_aware
            True
            >>> json_options = CANONICAL_JSON_OPTIONS.with_options(tz_aware=False, tzinfo=None)
            >>> json_options.tz_aware
            False

        .. versionadded:: 3.12
        """
        opts = self._options_dict()
        for opt in ('strict_number_long', 'datetime_representation', 'strict_uuid', 'json_mode'):
            opts[opt] = kwargs.get(opt, getattr(self, opt))
        opts.update(kwargs)
        return JSONOptions(**opts)