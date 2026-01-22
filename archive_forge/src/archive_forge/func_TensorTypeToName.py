import json
import os
import re
import sys
import numpy as np
def TensorTypeToName(tensor_type):
    """Converts a numerical enum to a readable tensor type."""
    for name, value in schema_fb.TensorType.__dict__.items():
        if value == tensor_type:
            return name
    return None