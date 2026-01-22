from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _file_content(entry):
    """Serialize the content component of entry which is a file.

    :param entry: An InventoryFile.
    """
    if entry.executable:
        exec_bytes = b'Y'
    else:
        exec_bytes = b''
    size_exec_sha = (entry.text_size, exec_bytes, entry.text_sha1)
    if None in size_exec_sha:
        raise InventoryDeltaError('Missing size or sha for %(fileid)r', fileid=entry.file_id)
    return b'file\x00%d\x00%s\x00%s' % size_exec_sha