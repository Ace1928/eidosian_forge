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
class StoredShape:
    """Normalized shape of image array in TIFF pages.

    Parameters:
        frames:
            Number of TIFF pages.
        separate_samples:
            Number of separate samples.
        depth:
            Image depth.
        length:
            Image length (height).
        width:
            Image width.
        contig_samples:
            Number of contiguous samples.
        extrasamples:
            Number of extra samples.

    """
    __slots__ = ('frames', 'separate_samples', 'depth', 'length', 'width', 'contig_samples', 'extrasamples')
    frames: int
    'Number of TIFF pages.'
    separate_samples: int
    'Number of separate samples.'
    depth: int
    'Image depth. Value of ImageDepth tag or 1.'
    length: int
    'Image length (height). Value of ImageLength tag.'
    width: int
    'Image width. Value of ImageWidth tag.'
    contig_samples: int
    'Number of contiguous samples.'
    extrasamples: int
    'Number of extra samples. Count of ExtraSamples tag or 0.'

    def __init__(self, frames: int=1, separate_samples: int=1, depth: int=1, length: int=1, width: int=1, contig_samples: int=1, extrasamples: int=0) -> None:
        if separate_samples != 1 and contig_samples != 1:
            raise ValueError('invalid samples')
        self.frames = int(frames)
        self.separate_samples = int(separate_samples)
        self.depth = int(depth)
        self.length = int(length)
        self.width = int(width)
        self.contig_samples = int(contig_samples)
        self.extrasamples = int(extrasamples)

    @property
    def size(self) -> int:
        """Product of all dimensions."""
        return abs(self.frames) * self.separate_samples * self.depth * self.length * self.width * self.contig_samples

    @property
    def samples(self) -> int:
        """Number of samples. Count of SamplesPerPixel tag."""
        assert self.separate_samples == 1 or self.contig_samples == 1
        samples = self.separate_samples if self.separate_samples > 1 else self.contig_samples
        assert self.extrasamples < samples
        return samples

    @property
    def photometric_samples(self) -> int:
        """Number of photometric samples."""
        return self.samples - self.extrasamples

    @property
    def shape(self) -> tuple[int, int, int, int, int, int]:
        """Normalized 6D shape of image array in all pages."""
        return (self.frames, self.separate_samples, self.depth, self.length, self.width, self.contig_samples)

    @property
    def page_shape(self) -> tuple[int, int, int, int, int]:
        """Normalized 5D shape of image array in single page."""
        return (self.separate_samples, self.depth, self.length, self.width, self.contig_samples)

    @property
    def page_size(self) -> int:
        """Product of dimensions in single page."""
        return self.separate_samples * self.depth * self.length * self.width * self.contig_samples

    @property
    def squeezed(self) -> tuple[int, ...]:
        """Shape with length-1 removed, except for length and width."""
        shape = [self.length, self.width]
        if self.separate_samples > 1:
            shape.insert(0, self.separate_samples)
        elif self.contig_samples > 1:
            shape.append(self.contig_samples)
        if self.frames > 1:
            shape.insert(0, self.frames)
        return tuple(shape)

    @property
    def is_valid(self) -> bool:
        """Shape is valid."""
        return self.frames >= 1 and self.depth >= 1 and (self.length >= 1) and (self.width >= 1) and (self.separate_samples == 1 or self.contig_samples == 1) and ((self.contig_samples if self.contig_samples > 1 else self.separate_samples) > self.extrasamples)

    @property
    def is_planar(self) -> bool:
        """Shape contains planar samples."""
        return self.separate_samples > 1

    @property
    def planarconfig(self) -> int | None:
        """Value of PlanarConfiguration tag."""
        if self.separate_samples > 1:
            return 2
        if self.contig_samples > 1:
            return 1
        return None

    def __len__(self) -> int:
        return 6

    @overload
    def __getitem__(self, key: int, /) -> int:
        ...

    @overload
    def __getitem__(self, key: slice, /) -> tuple[int, ...]:
        ...

    def __getitem__(self, key: int | slice, /) -> int | tuple[int, ...]:
        return (self.frames, self.separate_samples, self.depth, self.length, self.width, self.contig_samples)[key]

    def __eq__(self, other: object, /) -> bool:
        return isinstance(other, StoredShape) and self.frames == other.frames and (self.separate_samples == other.separate_samples) and (self.depth == other.depth) and (self.length == other.length) and (self.width == other.width) and (self.contig_samples == other.contig_samples)

    def __repr__(self) -> str:
        return f'<StoredShape(frames={self.frames}, separate_samples={self.separate_samples}, depth={self.depth}, length={self.length}, width={self.width}, contig_samples={self.contig_samples}, extrasamples={self.extrasamples})>'