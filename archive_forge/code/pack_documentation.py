import re
from io import BytesIO
from .. import errors
Take a line out of the buffer, and return the line.

        If a newline byte is not found in the buffer, the buffer is
        unchanged and this returns None instead.
        