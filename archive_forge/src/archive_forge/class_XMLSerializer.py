import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
class XMLSerializer(serializer.Serializer):
    """Abstract XML object serialize/deserialize"""
    squashes_xml_invalid_characters = True

    def read_inventory_from_lines(self, lines, revision_id=None, entry_cache=None, return_from_cache=False):
        """Read xml_string into an inventory object.

        :param chunks: The xml to read.
        :param revision_id: If not-None, the expected revision id of the
            inventory. Some serialisers use this to set the results' root
            revision. This should be supplied for deserialising all
            from-repository inventories so that xml5 inventories that were
            serialised without a revision identifier can be given the right
            revision id (but not for working tree inventories where users can
            edit the data without triggering checksum errors or anything).
        :param entry_cache: An optional cache of InventoryEntry objects. If
            supplied we will look up entries via (file_id, revision_id) which
            should map to a valid InventoryEntry (File/Directory/etc) object.
        :param return_from_cache: Return entries directly from the cache,
            rather than copying them first. This is only safe if the caller
            promises not to mutate the returned inventory entries, but it can
            make some operations significantly faster.
        """
        try:
            return self._unpack_inventory(fromstringlist(lines), revision_id, entry_cache=entry_cache, return_from_cache=return_from_cache)
        except ParseError as e:
            raise serializer.UnexpectedInventoryFormat(str(e))

    def read_inventory(self, f, revision_id=None):
        try:
            try:
                return self._unpack_inventory(self._read_element(f), revision_id=None)
            finally:
                f.close()
        except ParseError as e:
            raise serializer.UnexpectedInventoryFormat(str(e))

    def write_revision_to_string(self, rev):
        return b''.join(self.write_revision_to_lines(rev))

    def read_revision(self, f):
        return self._unpack_revision(self._read_element(f))

    def read_revision_from_string(self, xml_string):
        return self._unpack_revision(fromstring(xml_string))

    def _read_element(self, f):
        return ElementTree().parse(f)