from abc import ABC, abstractmethod
from .header import Field
class HeaderWarning(Warning):
    """Base class for warnings about tractogram file header."""