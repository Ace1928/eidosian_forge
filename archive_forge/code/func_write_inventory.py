from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
def write_inventory(self, inv, f, working=False):
    """Write inventory to a file.

        :param inv: the inventory to write.
        :param f: the file to write. (May be None if the lines are the desired
            output).
        :param working: If True skip history data - text_sha1, text_size,
            reference_revision, symlink_target.
        :return: The inventory as a list of lines.
        """
    output = []
    append = output.append
    self._append_inventory_root(append, inv)
    serialize_inventory_flat(inv, append, self.root_id, self.supported_kinds, working)
    if f is not None:
        f.writelines(output)
    return output