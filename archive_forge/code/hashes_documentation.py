from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl

        Return a HashContext that is a copy of the current context.
        