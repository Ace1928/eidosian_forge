import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
def unpack_inventory_flat(elt, format_num, unpack_entry, entry_cache=None, return_from_cache=False):
    """Unpack a flat XML inventory.

    :param elt: XML element for the inventory
    :param format_num: Expected format number
    :param unpack_entry: Function for unpacking inventory entries
    :return: An inventory
    :raise UnexpectedInventoryFormat: When unexpected elements or data is
        encountered
    """
    if elt.tag != 'inventory':
        raise serializer.UnexpectedInventoryFormat('Root tag is %r' % elt.tag)
    format = elt.get('format')
    if format is None and format_num is not None or format.encode() != format_num:
        raise serializer.UnexpectedInventoryFormat('Invalid format version %r' % format)
    revision_id = elt.get('revision_id')
    if revision_id is not None:
        revision_id = revision_id.encode('utf-8')
    inv = inventory.Inventory(root_id=None, revision_id=revision_id)
    for e in elt:
        ie = unpack_entry(e, entry_cache, return_from_cache)
        inv.add(ie)
    return inv