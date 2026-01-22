from abc import ABCMeta
from typing import Any, Dict, Optional, Pattern, Tuple, Type
import re

    Metaclass for creating XSD atomic types. The created classes
    are decorated with missing attributes and methods. When a name
    attribute is provided the class is registered into a global map
    of XSD atomic types and also the expanded name is added.
    