from io import BytesIO
from ase.io import iread, write
def _to_buffer(images, format=None, **kwargs):
    buf = BytesIO()
    write(buf, images, format=format, **kwargs)
    buf.seek(0)
    return buf