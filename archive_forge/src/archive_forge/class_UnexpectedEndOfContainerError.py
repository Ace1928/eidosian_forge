import re
from io import BytesIO
from .. import errors
class UnexpectedEndOfContainerError(ContainerError):
    _fmt = 'Unexpected end of container stream'