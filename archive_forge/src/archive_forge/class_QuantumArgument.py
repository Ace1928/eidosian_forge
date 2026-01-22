import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumArgument(QuantumDeclaration):
    """
    quantumArgument
        : 'qreg' Identifier designator? | 'qubit' designator? Identifier
    """