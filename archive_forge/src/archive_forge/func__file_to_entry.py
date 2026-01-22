from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _file_to_entry(content, name, parent_id, file_id, last_modified, _type=inventory.InventoryFile):
    """Convert a dir content record to an InventoryFile."""
    result = _type(file_id, name, parent_id)
    result.revision = last_modified
    result.text_size = int(content[1])
    result.text_sha1 = content[3]
    if content[2]:
        result.executable = True
    else:
        result.executable = False
    return result