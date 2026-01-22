from __future__ import annotations
from collections import namedtuple
from typing import Optional
Info about a driver wrapping PyMongo.

    The MongoDB server logs PyMongo's name, version, and platform whenever
    PyMongo establishes a connection. A driver implemented on top of PyMongo
    can add its own info to this log message. Initialize with three strings
    like 'MyDriver', '1.2.3', 'some platform info'. Any of these strings may be
    None to accept PyMongo's default.
    