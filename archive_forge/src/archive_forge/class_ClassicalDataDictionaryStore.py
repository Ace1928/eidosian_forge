import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
@value_equality_attr.value_equality(unhashable=True)
class ClassicalDataDictionaryStore(ClassicalDataStore):
    """Classical data representing measurements and metadata."""

    def __init__(self, *, _records: Optional[Dict['cirq.MeasurementKey', List[Tuple[int, ...]]]]=None, _measured_qubits: Optional[Dict['cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]]]=None, _channel_records: Optional[Dict['cirq.MeasurementKey', List[int]]]=None, _measurement_types: Optional[Dict['cirq.MeasurementKey', 'cirq.MeasurementType']]=None):
        """Initializes a `ClassicalDataDictionaryStore` object."""
        if not _measurement_types:
            _measurement_types = {}
            if _records:
                _measurement_types.update({k: MeasurementType.MEASUREMENT for k, v in _records.items()})
            if _channel_records:
                _measurement_types.update({k: MeasurementType.CHANNEL for k, v in _channel_records.items()})
        if _records is None:
            _records = {}
        if _measured_qubits is None:
            _measured_qubits = {}
        if _channel_records is None:
            _channel_records = {}
        self._records: Dict['cirq.MeasurementKey', List[Tuple[int, ...]]] = _records
        self._measured_qubits: Dict['cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]] = _measured_qubits
        self._channel_records: Dict['cirq.MeasurementKey', List[int]] = _channel_records
        self._measurement_types: Dict['cirq.MeasurementKey', 'cirq.MeasurementType'] = _measurement_types

    @property
    def records(self) -> Mapping['cirq.MeasurementKey', List[Tuple[int, ...]]]:
        """Gets the a mapping from measurement key to measurement records."""
        return self._records

    @property
    def channel_records(self) -> Mapping['cirq.MeasurementKey', List[int]]:
        """Gets the a mapping from measurement key to channel measurement records."""
        return self._channel_records

    @property
    def measured_qubits(self) -> Mapping['cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]]:
        """Gets the a mapping from measurement key to the qubits measured."""
        return self._measured_qubits

    @property
    def measurement_types(self) -> Mapping['cirq.MeasurementKey', 'cirq.MeasurementType']:
        """Gets the a mapping from measurement key to the measurement type."""
        return self._measurement_types

    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        return tuple(self._measurement_types.keys())

    def record_measurement(self, key: 'cirq.MeasurementKey', measurement: Sequence[int], qubits: Sequence['cirq.Qid']):
        if len(measurement) != len(qubits):
            raise ValueError(f'{len(measurement)} measurements but {len(qubits)} qubits.')
        if key not in self._measurement_types:
            self._measurement_types[key] = MeasurementType.MEASUREMENT
            self._records[key] = []
            self._measured_qubits[key] = []
        if self._measurement_types[key] != MeasurementType.MEASUREMENT:
            raise ValueError(f'Channel Measurement already logged to key {key}')
        measured_qubits = self._measured_qubits[key]
        if measured_qubits:
            shape = tuple((q.dimension for q in qubits))
            key_shape = tuple((q.dimension for q in measured_qubits[-1]))
            if shape != key_shape:
                raise ValueError(f'Measurement shape {shape} does not match {key_shape} in {key}.')
        measured_qubits.append(tuple(qubits))
        self._records[key].append(tuple(measurement))

    def record_channel_measurement(self, key: 'cirq.MeasurementKey', measurement: int):
        if key not in self._measurement_types:
            self._measurement_types[key] = MeasurementType.CHANNEL
            self._channel_records[key] = []
        if self._measurement_types[key] != MeasurementType.CHANNEL:
            raise ValueError(f'Measurement already logged to key {key}')
        self._channel_records[key].append(measurement)

    def get_digits(self, key: 'cirq.MeasurementKey', index=-1) -> Tuple[int, ...]:
        return self._records[key][index] if self._measurement_types[key] == MeasurementType.MEASUREMENT else (self._channel_records[key][index],)

    def get_int(self, key: 'cirq.MeasurementKey', index=-1) -> int:
        if key not in self._measurement_types:
            raise KeyError(f'The measurement key {key} is not in {self._measurement_types}')
        measurement_type = self._measurement_types[key]
        if measurement_type == MeasurementType.CHANNEL:
            return self._channel_records[key][index]
        if key not in self._measured_qubits:
            return digits.big_endian_bits_to_int(self._records[key][index])
        return digits.big_endian_digits_to_int(self._records[key][index], base=[q.dimension for q in self._measured_qubits[key][index]])

    def copy(self):
        return ClassicalDataDictionaryStore(_records=self._records.copy(), _measured_qubits=self._measured_qubits.copy(), _channel_records=self._channel_records.copy(), _measurement_types=self._measurement_types.copy())

    def _json_dict_(self):
        return {'records': list(self.records.items()), 'measured_qubits': list(self.measured_qubits.items()), 'channel_records': list(self.channel_records.items()), 'measurement_types': list(self.measurement_types.items())}

    @classmethod
    def _from_json_dict_(cls, records, measured_qubits, channel_records, measurement_types, **kwargs):
        return cls(_records=dict(records), _measured_qubits=dict(measured_qubits), _channel_records=dict(channel_records), _measurement_types=dict(measurement_types))

    def __repr__(self):
        rep = 'cirq.ClassicalDataDictionaryStore('
        if self.records:
            rep += f'_records={self.records!r},'
        if self.measured_qubits:
            rep += f' _measured_qubits={self.measured_qubits!r},'
        if self.channel_records:
            rep += f' _channel_records={self.channel_records!r},'
        if self.measurement_types:
            rep += f' _measurement_types={self.measurement_types!r},'
        return rep + ')'

    def _value_equality_values_(self):
        return (self._records, self._channel_records, self._measurement_types, self._measured_qubits)