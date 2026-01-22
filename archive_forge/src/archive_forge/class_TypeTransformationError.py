import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class TypeTransformationError(Error):
    """Error transforming between python proto type and corresponding C++ type."""