from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
class Serializer_v8(XMLSerializer):
    """This serialiser adds rich roots.

    Its revision format number matches its inventory number.
    """
    __slots__: List[str] = []
    root_id: Optional[bytes] = None
    support_altered_by_hack = True
    supported_kinds = {'file', 'directory', 'symlink'}
    format_num = b'8'
    revision_format_num: Optional[bytes] = None
    _file_ids_altered_regex = lazy_regex.lazy_compile(b'file_id="(?P<file_id>[^"]+)".* revision="(?P<revision_id>[^"]+)"')

    def _check_revisions(self, inv):
        """Extension point for subclasses to check during serialisation.

        :param inv: An inventory about to be serialised, to be checked.
        :raises: AssertionError if an error has occurred.
        """
        if inv.revision_id is None:
            raise AssertionError('inv.revision_id is None')
        if inv.root.revision is None:
            raise AssertionError('inv.root.revision is None')

    def _check_cache_size(self, inv_size, entry_cache):
        """Check that the entry_cache is large enough.

        We want the cache to be ~2x the size of an inventory. The reason is
        because we use a FIFO cache, and how Inventory records are likely to
        change. In general, you have a small number of records which change
        often, and a lot of records which do not change at all. So when the
        cache gets full, you actually flush out a lot of the records you are
        interested in, which means you need to recreate all of those records.
        An LRU Cache would be better, but the overhead negates the cache
        coherency benefit.

        One way to look at it, only the size of the cache > len(inv) is your
        'working' set. And in general, it shouldn't be a problem to hold 2
        inventories in memory anyway.

        :param inv_size: The number of entries in an inventory.
        """
        if entry_cache is None:
            return
        recommended_min_cache_size = inv_size * 1.5
        if entry_cache.cache_size() < recommended_min_cache_size:
            recommended_cache_size = inv_size * 2
            trace.mutter('Resizing the inventory entry cache from %d to %d', entry_cache.cache_size(), recommended_cache_size)
            entry_cache.resize(recommended_cache_size)

    def write_inventory_to_lines(self, inv):
        """Return a list of lines with the encoded inventory."""
        return self.write_inventory(inv, None)

    def write_inventory_to_chunks(self, inv):
        return self.write_inventory(inv, None)

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

    def _append_inventory_root(self, append, inv):
        """Append the inventory root to output."""
        if inv.revision_id is not None:
            revid1 = b''.join([b' revision_id="', encode_and_escape(inv.revision_id), b'"'])
        else:
            revid1 = b''
        append(b'<inventory format="%s"%s>\n' % (self.format_num, revid1))
        append(b'<directory file_id="%s" name="%s" revision="%s" />\n' % (encode_and_escape(inv.root.file_id), encode_and_escape(inv.root.name), encode_and_escape(inv.root.revision)))

    def write_revision_to_lines(self, rev):
        """Revision object -> xml tree"""
        lines = []
        el = b'<revision committer="%s" format="%s" inventory_sha1="%s" revision_id="%s" timestamp="%.3f"' % (encode_and_escape(rev.committer), self.revision_format_num or self.format_num, rev.inventory_sha1, encode_and_escape(rev.revision_id.decode('utf-8')), rev.timestamp)
        if rev.timezone is not None:
            el += b' timezone="%s"' % str(rev.timezone).encode('ascii')
        lines.append(el + b'>\n')
        message = encode_and_escape(escape_invalid_chars(rev.message)[0])
        lines.extend((b'<message>' + message + b'</message>\n').splitlines(True))
        if rev.parent_ids:
            lines.append(b'<parents>\n')
            for parent_id in rev.parent_ids:
                _mod_revision.check_not_reserved_id(parent_id)
                lines.append(b'<revision_ref revision_id="%s" />\n' % encode_and_escape(parent_id.decode('utf-8')))
            lines.append(b'</parents>\n')
        if rev.properties:
            preamble = b'<properties>'
            for prop_name, prop_value in sorted(rev.properties.items()):
                if prop_value:
                    proplines = (preamble + b'<property name="%s">%s</property>\n' % (encode_and_escape(prop_name), encode_and_escape(escape_invalid_chars(prop_value)[0]))).splitlines(True)
                else:
                    proplines = [preamble + b'<property name="%s" />\n' % (encode_and_escape(prop_name),)]
                preamble = b''
                lines.extend(proplines)
            lines.append(b'</properties>\n')
        lines.append(b'</revision>\n')
        return lines

    def _unpack_entry(self, elt, entry_cache=None, return_from_cache=False):
        return unpack_inventory_entry(elt, entry_cache, return_from_cache)

    def _unpack_inventory(self, elt, revision_id=None, entry_cache=None, return_from_cache=False):
        """Construct from XML Element"""
        inv = unpack_inventory_flat(elt, self.format_num, self._unpack_entry, entry_cache, return_from_cache)
        self._check_cache_size(len(inv), entry_cache)
        return inv

    def _unpack_revision(self, elt):
        """XML Element -> Revision object"""
        format = elt.get('format')
        format_num = self.format_num
        if self.revision_format_num is not None:
            format_num = self.revision_format_num
        if format is not None:
            if format.encode() != format_num:
                raise BzrError('invalid format version %r on revision' % format)
        get_cached = get_utf8_or_ascii
        rev = Revision(committer=elt.get('committer'), timestamp=float(elt.get('timestamp')), revision_id=get_cached(elt.get('revision_id')), inventory_sha1=elt.get('inventory_sha1').encode('ascii'))
        parents = elt.find('parents')
        if parents is not None:
            for p in parents:
                rev.parent_ids.append(get_cached(p.get('revision_id')))
        self._unpack_revision_properties(elt, rev)
        v = elt.get('timezone')
        if v is None:
            rev.timezone = 0
        else:
            rev.timezone = int(v)
        rev.message = elt.findtext('message')
        return rev

    def _unpack_revision_properties(self, elt, rev):
        """Unpack properties onto a revision."""
        props_elt = elt.find('properties')
        if props_elt is None:
            return
        for prop_elt in props_elt:
            if prop_elt.tag != 'property':
                raise AssertionError('bad tag under properties list: %r' % prop_elt.tag)
            name = prop_elt.get('name')
            value = prop_elt.text
            if value is None:
                value = ''
            if name in rev.properties:
                raise AssertionError('repeated property %r' % name)
            rev.properties[name] = value

    def _find_text_key_references(self, line_iterator):
        """Core routine for extracting references to texts from inventories.

        This performs the translation of xml lines to revision ids.

        :param line_iterator: An iterator of lines, origin_version_id
        :return: A dictionary mapping text keys ((fileid, revision_id) tuples)
            to whether they were referred to by the inventory of the
            revision_id that they contain. Note that if that revision_id was
            not part of the line_iterator's output then False will be given -
            even though it may actually refer to that key.
        """
        if not self.support_altered_by_hack:
            raise AssertionError('_find_text_key_references only supported for branches which store inventory as unnested xml, not on %r' % self)
        result = {}
        unescape_revid_cache = {}
        unescape_fileid_cache = {}
        search = self._file_ids_altered_regex.search
        unescape = _unescape_xml
        setdefault = result.setdefault
        for line, line_key in line_iterator:
            match = search(line)
            if match is None:
                continue
            file_id, revision_id = match.group('file_id', 'revision_id')
            try:
                revision_id = unescape_revid_cache[revision_id]
            except KeyError:
                unescaped = unescape(revision_id)
                unescape_revid_cache[revision_id] = unescaped
                revision_id = unescaped
            try:
                file_id = unescape_fileid_cache[file_id]
            except KeyError:
                unescaped = unescape(file_id)
                unescape_fileid_cache[file_id] = unescaped
                file_id = unescaped
            key = (file_id, revision_id)
            setdefault(key, False)
            if revision_id == line_key[-1]:
                result[key] = True
        return result