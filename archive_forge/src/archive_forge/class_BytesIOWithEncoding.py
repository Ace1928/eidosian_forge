import io
from .. import ui
from ..ui import text as ui_text
class BytesIOWithEncoding(io.BytesIO):
    encoding = 'ascii'