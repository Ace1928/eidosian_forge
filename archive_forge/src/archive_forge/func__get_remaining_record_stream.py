import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def _get_remaining_record_stream(self, keys, ordering, include_delta_closure):
    """This function is the 'retry' portion for get_record_stream."""
    if include_delta_closure:
        positions = self._get_components_positions(keys, allow_missing=True)
    else:
        build_details = self._index.get_build_details(keys)
        positions = {key: self._build_details_to_components(details) for key, details in build_details.items()}
    absent_keys = keys.difference(set(positions))
    if include_delta_closure:
        needed_from_fallback = set()
        reconstructable_keys = {}
        for key in keys:
            try:
                chain = [key, positions[key][2]]
            except KeyError:
                needed_from_fallback.add(key)
                continue
            result = True
            while chain[-1] is not None:
                if chain[-1] in reconstructable_keys:
                    result = reconstructable_keys[chain[-1]]
                    break
                else:
                    try:
                        chain.append(positions[chain[-1]][2])
                    except KeyError:
                        needed_from_fallback.add(chain[-1])
                        result = True
                        break
            for chain_key in chain[:-1]:
                reconstructable_keys[chain_key] = result
            if not result:
                needed_from_fallback.add(key)
    global_map, parent_maps = self._get_parent_map_with_sources(keys)
    if ordering in ('topological', 'groupcompress'):
        if ordering == 'topological':
            present_keys = tsort.topo_sort(global_map)
        else:
            present_keys = sort_groupcompress(global_map)
        source_keys = []
        current_source = None
        for key in present_keys:
            for parent_map in parent_maps:
                if key in parent_map:
                    key_source = parent_map
                    break
            if current_source is not key_source:
                source_keys.append((key_source, []))
                current_source = key_source
            source_keys[-1][1].append(key)
    else:
        if ordering != 'unordered':
            raise AssertionError('valid values for ordering are: "unordered", "groupcompress" or "topological" not: %r' % (ordering,))
        present_keys = []
        source_keys = []
        for parent_map in reversed(parent_maps):
            source_keys.append((parent_map, []))
            for key in parent_map:
                present_keys.append(key)
                source_keys[-1][1].append(key)
        for source, sub_keys in source_keys:
            if source is parent_maps[0]:
                self._index._sort_keys_by_io(sub_keys, positions)
    absent_keys = keys - set(global_map)
    for key in absent_keys:
        yield AbsentContentFactory(key)
    if include_delta_closure:
        non_local_keys = needed_from_fallback - absent_keys
        for keys, non_local_keys in self._group_keys_for_io(present_keys, non_local_keys, positions):
            generator = _VFContentMapGenerator(self, keys, non_local_keys, global_map, ordering=ordering)
            yield from generator.get_record_stream()
    else:
        for source, keys in source_keys:
            if source is parent_maps[0]:
                records = [(key, positions[key][1]) for key in keys]
                for key, raw_data in self._read_records_iter_unchecked(records):
                    record_details, index_memo, _ = positions[key]
                    yield KnitContentFactory(key, global_map[key], record_details, None, raw_data, self._factory.annotated, None)
            else:
                vf = self._immediate_fallback_vfs[parent_maps.index(source) - 1]
                yield from vf.get_record_stream(keys, ordering, include_delta_closure)