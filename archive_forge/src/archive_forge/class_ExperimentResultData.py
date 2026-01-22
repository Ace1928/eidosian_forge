import copy
from qiskit.qobj.utils import MeasReturnType, MeasLevel
from qiskit.qobj import QobjExperimentHeader
from qiskit.exceptions import QiskitError
class ExperimentResultData:
    """Class representing experiment result data"""

    def __init__(self, counts=None, snapshots=None, memory=None, statevector=None, unitary=None, **kwargs):
        """Initialize an ExperimentalResult Data class

        Args:
            counts (dict): A dictionary where the keys are the result in
                hexadecimal as string of the format "0xff" and the value
                is the number of counts for that result
            snapshots (dict): A dictionary where the key is the snapshot
                slot and the value is a dictionary of the snapshots for
                that slot.
            memory (list): A list of results per shot if the run had
                memory enabled
            statevector (list or numpy.array): A list or numpy array of the
                statevector result
            unitary (list or numpy.array): A list or numpy array of the
                unitary result
            kwargs (any): additional data key-value pairs.
        """
        self._data_attributes = []
        if counts is not None:
            self._data_attributes.append('counts')
            self.counts = counts
        if snapshots is not None:
            self._data_attributes.append('snapshots')
            self.snapshots = snapshots
        if memory is not None:
            self._data_attributes.append('memory')
            self.memory = memory
        if statevector is not None:
            self._data_attributes.append('statevector')
            self.statevector = statevector
        if unitary is not None:
            self._data_attributes.append('unitary')
            self.unitary = unitary
        for key, value in kwargs.items():
            setattr(self, key, value)
            self._data_attributes.append(key)

    def __repr__(self):
        string_list = []
        for field in self._data_attributes:
            string_list.append(f'{field}={getattr(self, field)}')
        out = 'ExperimentResultData(%s)' % ', '.join(string_list)
        return out

    def to_dict(self):
        """Return a dictionary format representation of the ExperimentResultData

        Returns:
            dict: The dictionary form of the ExperimentResultData
        """
        out_dict = {}
        for field in self._data_attributes:
            out_dict[field] = getattr(self, field)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new ExperimentResultData object from a dictionary.

        Args:
            data (dict): A dictionary representing the ExperimentResultData to
                         create. It will be in the same format as output by
                         :meth:`to_dict`
        Returns:
            ExperimentResultData: The ``ExperimentResultData`` object from the
                                  input dictionary.
        """
        in_data = copy.copy(data)
        return cls(**in_data)