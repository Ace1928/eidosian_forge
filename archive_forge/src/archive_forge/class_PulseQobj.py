import copy
import pprint
from typing import Union, List
import numpy
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader
class PulseQobj:
    """A Pulse Qobj."""

    def __init__(self, qobj_id, config, experiments, header=None):
        """Instantiate a new Pulse Qobj Object.

        Each Pulse Qobj object is used to represent a single payload that will
        be passed to a Qiskit provider. It mirrors the Qobj the published
        `Qobj specification <https://arxiv.org/abs/1809.03452>`_ for Pulse
        experiments.

        Args:
            qobj_id (str): An identifier for the qobj
            config (PulseQobjConfig): A config for the entire run
            header (QobjHeader): A header for the entire run
            experiments (list): A list of lists of :class:`PulseQobjExperiment`
                objects representing an experiment
        """
        self.qobj_id = qobj_id
        self.config = config
        self.header = header or QobjHeader()
        self.experiments = experiments
        self.type = 'PULSE'
        self.schema_version = '1.2.0'

    def __repr__(self):
        experiments_str = [repr(x) for x in self.experiments]
        experiments_repr = '[' + ', '.join(experiments_str) + ']'
        out = "PulseQobj(qobj_id='{}', config={}, experiments={}, header={})".format(self.qobj_id, repr(self.config), experiments_repr, repr(self.header))
        return out

    def __str__(self):
        out = 'Pulse Qobj: %s:\n' % self.qobj_id
        config = pprint.pformat(self.config.to_dict())
        out += 'Config: %s\n' % str(config)
        header = pprint.pformat(self.header.to_dict())
        out += 'Header: %s\n' % str(header)
        out += 'Experiments:\n'
        for experiment in self.experiments:
            out += '%s' % str(experiment)
        return out

    def to_dict(self):
        """Return a dictionary format representation of the Pulse Qobj.

        Note this dict is not in the json wire format expected by IBMQ and qobj
        specification because complex numbers are still of type complex. Also
        this may contain native numpy arrays. When serializing this output
        for use with IBMQ you can leverage a json encoder that converts these
        as expected. For example:

        .. code-block::

            import json
            import numpy

            class QobjEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, numpy.ndarray):
                        return obj.tolist()
                    if isinstance(obj, complex):
                        return (obj.real, obj.imag)
                    return json.JSONEncoder.default(self, obj)

            json.dumps(qobj.to_dict(), cls=QobjEncoder)

        Returns:
            dict: A dictionary representation of the PulseQobj object
        """
        out_dict = {'qobj_id': self.qobj_id, 'header': self.header.to_dict(), 'config': self.config.to_dict(), 'schema_version': self.schema_version, 'type': self.type, 'experiments': [x.to_dict() for x in self.experiments]}
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseQobj object from a dictionary.

        Args:
            data (dict): A dictionary representing the PulseQobj to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            PulseQobj: The PulseQobj from the input dictionary.
        """
        config = None
        if 'config' in data:
            config = PulseQobjConfig.from_dict(data['config'])
        experiments = None
        if 'experiments' in data:
            experiments = [PulseQobjExperiment.from_dict(exp) for exp in data['experiments']]
        header = None
        if 'header' in data:
            header = QobjHeader.from_dict(data['header'])
        return cls(qobj_id=data.get('qobj_id'), config=config, experiments=experiments, header=header)

    def __eq__(self, other):
        if isinstance(other, PulseQobj):
            if self.to_dict() == other.to_dict():
                return True
        return False