import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def _register_resolver(dict_factory: Callable[[], Dict[str, ObjectFactory]]) -> None:
    """Register a resolver based on a dict factory for lazy initialization.

    Cirq modules are the ones referred in cirq/__init__.py. If a Cirq module
    wants to expose JSON serializable objects, it should register itself using
    this method to be supported by the protocol. See for example
    cirq/__init__.py or cirq/google/__init__.py.

    As Cirq modules are imported by cirq/__init__.py, they are different from
    3rd party packages, and as such SHOULD NEVER rely on storing a
    separate resolver based on DEAFULT_RESOLVERS because that will cause a
    partial DEFAULT_RESOLVER to be used by that module. What it contains will
    depend on where in cirq/__init__.py the module is imported first, as some
    modules might not had the chance to register themselves yet.

    Args:
        dict_factory: the callable that returns the actual dict for type names
            to types (ObjectFactory)
    """
    DEFAULT_RESOLVERS.append(_lazy_resolver(dict_factory))