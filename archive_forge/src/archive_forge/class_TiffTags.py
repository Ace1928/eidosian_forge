from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@final
class TiffTags:
    """Multidict-like interface to TiffTag instances in TiffPage.

    Differences to a regular dict:

    - values are instances of :py:class:`TiffTag`.
    - keys are :py:attr:`TiffTag.code` (int).
    - multiple values can be stored per key.
    - can be indexed by :py:attr:`TiffTag.name` (`str`), slower than by key.
    - `iter()` returns values instead of keys.
    - `values()` and `items()` contain all values sorted by offset.
    - `len()` returns number of all values.
    - `get()` takes optional index argument.
    - some functions are not implemented, such as, `update` and `pop`.

    """
    __slots__ = ('_dict', '_list')
    _dict: dict[int, TiffTag]
    _list: list[dict[int, TiffTag]]

    def __init__(self) -> None:
        self._dict = {}
        self._list = [self._dict]

    def add(self, tag: TiffTag, /) -> None:
        """Add tag."""
        code = tag.code
        for d in self._list:
            if code not in d:
                d[code] = tag
                break
        else:
            self._list.append({code: tag})

    def keys(self) -> list[int]:
        """Return codes of all tags."""
        return list(self._dict.keys())

    def values(self) -> list[TiffTag]:
        """Return all tags in order they are stored in file."""
        tags = (t for d in self._list for t in d.values())
        return sorted(tags, key=lambda t: t.offset)

    def items(self) -> list[tuple[int, TiffTag]]:
        """Return all (code, tag) pairs in order tags are stored in file."""
        items = (i for d in self._list for i in d.items())
        return sorted(items, key=lambda i: i[1].offset)

    def valueof(self, key: int | str, /, default: Any=None, index: int | None=None) -> Any:
        """Return value of tag by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tag to return.
            default:
                Another value to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        tag = self.get(key, default=None, index=index)
        if tag is None:
            return default
        try:
            return tag.value
        except TiffFileError:
            return default

    def get(self, key: int | str, /, default: TiffTag | None=None, index: int | None=None) -> TiffTag | None:
        """Return tag by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tag to return.
            default:
                Another tag to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        if index is None:
            if key in self._dict:
                return self._dict[cast(int, key)]
            if not isinstance(key, str):
                return default
            index = 0
        try:
            tags = self._list[index]
        except IndexError:
            return default
        if key in tags:
            return tags[cast(int, key)]
        if not isinstance(key, str):
            return default
        for tag in tags.values():
            if tag.name == key:
                return tag
        return default

    def getall(self, key: int | str, /, default=None) -> list[TiffTag] | None:
        """Return list of all tags by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tags to return.
            default:
                Value to return if no tags are found.

        """
        result: list[TiffTag] = []
        for tags in self._list:
            if key in tags:
                result.append(tags[cast(int, key)])
            else:
                break
        if result:
            return result
        if not isinstance(key, str):
            return default
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    result.append(tag)
                    break
            if not result:
                break
        return result if result else default

    def __getitem__(self, key: int | str, /) -> TiffTag:
        """Return first tag by code or name. Raise KeyError if not found."""
        if key in self._dict:
            return self._dict[cast(int, key)]
        if not isinstance(key, str):
            raise KeyError(key)
        for tag in self._dict.values():
            if tag.name == key:
                return tag
        raise KeyError(key)

    def __setitem__(self, code: int, tag: TiffTag, /) -> None:
        """Add tag."""
        assert tag.code == code
        self.add(tag)

    def __delitem__(self, key: int | str, /) -> None:
        """Delete all tags by code or name."""
        found = False
        for tags in self._list:
            if key in tags:
                found = True
                del tags[cast(int, key)]
            else:
                break
        if found:
            return
        if not isinstance(key, str):
            raise KeyError(key)
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    del tags[tag.code]
                    found = True
                    break
            else:
                break
        if not found:
            raise KeyError(key)
        return

    def __contains__(self, item: object, /) -> bool:
        """Return if tag is in map."""
        if item in self._dict:
            return True
        if not isinstance(item, str):
            return False
        for tag in self._dict.values():
            if tag.name == item:
                return True
        return False

    def __iter__(self) -> Iterator[TiffTag]:
        """Return iterator over all tags."""
        return iter(self.values())

    def __len__(self) -> int:
        """Return number of tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size

    def __repr__(self) -> str:
        return f'<tifffile.TiffTags @0x{id(self):016X}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int=0, width: int=79) -> str:
        """Return string with information about TiffTags."""
        info = []
        tlines = []
        vlines = []
        for tag in self:
            value = tag._str(width=width + 1)
            tlines.append(value[:width].strip())
            if detail > 0 and len(value) > width:
                try:
                    value = tag.value
                except Exception:
                    continue
                if tag.code in {273, 279, 324, 325}:
                    if detail < 1:
                        value = value[:256]
                    elif len(value) > 1024:
                        value = value[:512] + value[-512:]
                    value = pformat(value, width=width, height=detail * 3)
                else:
                    value = pformat(value, width=width, height=detail * 8)
                if tag.count > 1:
                    vlines.append(f'{tag.name} {tag.dtype_name}[{tag.count}]\n{value}')
                else:
                    vlines.append(f'{tag.name}\n{value}')
        info.append('\n'.join(tlines))
        if detail > 0 and vlines:
            info.append('\n')
            info.append('\n\n'.join(vlines))
        return '\n'.join(info)