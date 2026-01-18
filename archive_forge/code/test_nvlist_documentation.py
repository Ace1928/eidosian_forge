import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (

Tests for _nvlist module.
The tests convert from a `dict` to C ``nvlist_t`` and back to a `dict`
and verify that no information is lost and value types are correct.
The tests also check that various error conditions like unsupported
value types or out of bounds values are detected.
