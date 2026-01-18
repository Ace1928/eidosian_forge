from enum import Enum
from typing import List, Dict
from qiskit.circuit.library.templates import rzx
Convenience function to get the cost_dict and templates for template matching.

    Args:
        template_list: List of instruction names.

    Returns:
        Decomposition templates and cost values.
    