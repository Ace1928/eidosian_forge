import re
import traitlets
import datetime as dt
class ByteMemoryView(traitlets.TraitType):
    """A trait for memory views of bytes."""
    default_value = memoryview(b'')
    info_text = 'a memory view object'

    def validate(self, obj, value):
        if isinstance(value, memoryview) and value.format == 'B':
            return value
        self.error(obj, value)

    def default_value_repr(self):
        return repr(self.default_value.tobytes())