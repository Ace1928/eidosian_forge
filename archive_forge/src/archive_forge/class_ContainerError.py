import re
from io import BytesIO
from .. import errors
class ContainerError(errors.BzrError):
    """Base class of container errors."""