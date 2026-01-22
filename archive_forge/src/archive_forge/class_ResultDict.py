import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
class ResultDict(Result):
    """A Result created from a dict mapping measurement keys to measured values.

    Stores results of executing a circuit for multiple repetitions with one
    fixed set of parameters. The values for each measurement key are stored as a
    2D numpy array. The first (row) index in each array is the repetition
    number, and the second (column) index is the qubit.

    Attributes:
        params: A ParamResolver of settings used when sampling result.
    """

    def __init__(self, *, params: Optional[resolver.ParamResolver]=None, measurements: Optional[Mapping[str, np.ndarray]]=None, records: Optional[Mapping[str, np.ndarray]]=None) -> None:
        """Inits Result.

        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. The value for each key is a 2-D array of booleans,
                with the first index running over the repetitions, and the
                second index running over the qubits for the corresponding
                measurements.
            records: A dictionary from measurement gate key to measurement
                results. The value for each key is a 3D array of booleans,
                with the first index running over the repetitions, the second
                index running over "instances" of that key in the circuit, and
                the last index running over the qubits for the corresponding
                measurements.
        """
        if params is None:
            params = resolver.ParamResolver({})
        if measurements is None and records is None:
            measurements = {}
            records = {}
        self._params = params
        self._measurements = measurements
        self._records = records
        self._data: Optional[pd.DataFrame] = None

    @property
    def params(self) -> 'cirq.ParamResolver':
        return self._params

    @property
    def measurements(self) -> Mapping[str, np.ndarray]:
        if self._measurements is None:
            assert self._records is not None
            self._measurements = {}
            for key, data in self._records.items():
                reps, instances, qubits = data.shape
                if instances != 1:
                    raise ValueError('Cannot extract 2D measurements for repeated keys')
                self._measurements[key] = data.reshape((reps, qubits))
        return self._measurements

    @property
    def records(self) -> Mapping[str, np.ndarray]:
        if self._records is None:
            assert self._measurements is not None
            self._records = {key: data[:, np.newaxis, :] for key, data in self._measurements.items()}
        return self._records

    @property
    def repetitions(self) -> int:
        if self._records is not None:
            if not self._records:
                return 0
            return len(next(iter(self._records.values())))
        else:
            if not self._measurements:
                return 0
            return len(next(iter(self._measurements.values())))

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self.dataframe_from_measurements(self.measurements)
        return self._data

    def _record_dict_repr(self):
        """Helper function for use in __repr__ to display the records field."""
        return '{' + ', '.join((f'{k!r}: {proper_repr(v)}' for k, v in self.records.items())) + '}'

    def __repr__(self) -> str:
        return f'cirq.ResultDict(params={self.params!r}, records={self._record_dict_repr()})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Output to show in ipython and Jupyter notebooks."""
        if cycle:
            p.text('ResultDict(...)')
        else:
            p.text(str(self))

    def __str__(self) -> str:
        return _keyed_repeated_bitstrings(self.measurements)

    def _json_dict_(self):
        packed_records = {}
        for key, digits in self.records.items():
            packed_digits, binary = _pack_digits(digits)
            packed_records[key] = {'packed_digits': packed_digits, 'binary': binary, 'dtype': digits.dtype.name, 'shape': digits.shape}
        return {'params': self.params, 'records': packed_records}

    @classmethod
    def _from_packed_records(cls, records, **kwargs):
        """Helper function for `_from_json_dict_` to construct from packed records."""
        return cls(records={key: _unpack_digits(**val) for key, val in records.items()}, **kwargs)

    @classmethod
    def _from_json_dict_(cls, params, **kwargs):
        if 'measurements' in kwargs:
            measurements = kwargs['measurements']
            return cls(params=params, measurements={key: _unpack_digits(**val) for key, val in measurements.items()})
        return cls._from_packed_records(params=params, records=kwargs['records'])