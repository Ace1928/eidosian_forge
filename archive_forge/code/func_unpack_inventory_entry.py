import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
def unpack_inventory_entry(elt, entry_cache=None, return_from_cache=False):
    elt_get = elt.get
    file_id = elt_get('file_id')
    revision = elt_get('revision')
    if entry_cache is not None and revision is not None:
        key = (file_id, revision)
        try:
            cached_ie = entry_cache[key]
        except KeyError:
            pass
        else:
            if return_from_cache:
                if cached_ie.kind == 'directory':
                    return cached_ie.copy()
                return cached_ie
            return cached_ie.copy()
    kind = elt.tag
    if not inventory.InventoryEntry.versionable_kind(kind):
        raise AssertionError('unsupported entry kind %s' % kind)
    file_id = get_utf8_or_ascii(file_id)
    if revision is not None:
        revision = get_utf8_or_ascii(revision)
    parent_id = elt_get('parent_id')
    if parent_id is not None:
        parent_id = get_utf8_or_ascii(parent_id)
    if kind == 'directory':
        ie = inventory.InventoryDirectory(file_id, elt_get('name'), parent_id)
    elif kind == 'file':
        ie = inventory.InventoryFile(file_id, elt_get('name'), parent_id)
        ie.text_sha1 = elt_get('text_sha1')
        if ie.text_sha1 is not None:
            ie.text_sha1 = ie.text_sha1.encode('ascii')
        if elt_get('executable') == 'yes':
            ie.executable = True
        v = elt_get('text_size')
        ie.text_size = v and int(v)
    elif kind == 'symlink':
        ie = inventory.InventoryLink(file_id, elt_get('name'), parent_id)
        ie.symlink_target = elt_get('symlink_target')
    elif kind == 'tree-reference':
        file_id = get_utf8_or_ascii(elt.attrib['file_id'])
        name = elt.attrib['name']
        parent_id = get_utf8_or_ascii(elt.attrib['parent_id'])
        revision = get_utf8_or_ascii(elt.get('revision'))
        reference_revision = get_utf8_or_ascii(elt.get('reference_revision'))
        ie = inventory.TreeReference(file_id, name, parent_id, revision, reference_revision)
    else:
        raise serializer.UnsupportedInventoryKind(kind)
    ie.revision = revision
    if revision is not None and entry_cache is not None:
        entry_cache[key] = ie.copy()
    return ie