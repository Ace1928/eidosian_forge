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
class CirqEncoder(json.JSONEncoder):
    """Extend json.JSONEncoder to support Cirq objects.

    This supports custom serialization. For details, see the documentation
    for the SupportsJSON protocol.

    In addition to serializing objects that implement the SupportsJSON
    protocol, this encoder deals with common, basic types:

     - Python complex numbers get saved as a dictionary keyed by 'real'
       and 'imag'.
     - Numpy ndarrays are converted to lists to use the json module's
       built-in support for lists.
     - Preliminary support for Sympy objects. Currently only sympy.Symbol.
       See https://github.com/quantumlib/Cirq/issues/2014
    """

    def default(self, o):
        if hasattr(o, '_json_dict_'):
            return _json_dict_with_cirq_type(o)
        if isinstance(o, sympy.Symbol):
            return {'cirq_type': 'sympy.Symbol', 'name': o.name}
        if isinstance(o, (sympy.Add, sympy.Mul, sympy.Pow, sympy.GreaterThan, sympy.StrictGreaterThan, sympy.LessThan, sympy.StrictLessThan, sympy.Equality, sympy.Unequality)):
            return {'cirq_type': f'sympy.{o.__class__.__name__}', 'args': o.args}
        if isinstance(o, sympy.Integer):
            return {'cirq_type': 'sympy.Integer', 'i': o.p}
        if isinstance(o, sympy.Float):
            return {'cirq_type': 'sympy.Float', 'approx': float(o)}
        if isinstance(o, sympy.Rational):
            return {'cirq_type': 'sympy.Rational', 'p': o.p, 'q': o.q}
        if isinstance(o, sympy.NumberSymbol):
            if o is sympy.pi:
                return {'cirq_type': 'sympy.pi'}
            if o is sympy.E:
                return {'cirq_type': 'sympy.E'}
            if o is sympy.EulerGamma:
                return {'cirq_type': 'sympy.EulerGamma'}
        if isinstance(o, numbers.Integral):
            return int(o)
        if isinstance(o, numbers.Real):
            return float(o)
        if isinstance(o, numbers.Complex):
            return {'cirq_type': 'complex', 'real': o.real, 'imag': o.imag}
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.MultiIndex):
            return {'cirq_type': 'pandas.MultiIndex', 'tuples': list(o), 'names': list(o.names)}
        if isinstance(o, pd.Index):
            return {'cirq_type': 'pandas.Index', 'data': list(o), 'name': o.name}
        if isinstance(o, pd.DataFrame):
            cols = [o[col].tolist() for col in o.columns]
            rows = list(zip(*cols))
            return {'cirq_type': 'pandas.DataFrame', 'data': rows, 'columns': o.columns, 'index': o.index}
        if isinstance(o, datetime.datetime):
            return {'cirq_type': 'datetime.datetime', 'timestamp': o.timestamp()}
        return super().default(o)