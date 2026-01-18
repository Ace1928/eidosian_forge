from typing import TYPE_CHECKING
from ..cloudpath import CloudPath, NoStatError
Abstract CloudPath for accessing objects the local filesystem. Subclasses are as a
    monkeypatch substitutes for normal CloudPath subclasses when writing tests.