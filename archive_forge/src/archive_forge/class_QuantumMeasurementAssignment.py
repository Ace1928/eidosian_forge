import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumMeasurementAssignment(Statement):
    """
    quantumMeasurementAssignment
        : quantumMeasurement ARROW indexIdentifierList
        | indexIdentifier EQUALS quantumMeasurement  # eg: bits = measure qubits;
    """

    def __init__(self, identifier: Identifier, quantumMeasurement: QuantumMeasurement):
        self.identifier = identifier
        self.quantumMeasurement = quantumMeasurement