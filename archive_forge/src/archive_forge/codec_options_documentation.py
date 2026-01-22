from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
Make a copy of this CodecOptions, overriding some options::

                >>> from bson.codec_options import DEFAULT_CODEC_OPTIONS
                >>> DEFAULT_CODEC_OPTIONS.tz_aware
                False
                >>> options = DEFAULT_CODEC_OPTIONS.with_options(tz_aware=True)
                >>> options.tz_aware
                True

            .. versionadded:: 3.5
            