from . import errors, osutils
def _walk_others(self):
    """Finish up by walking all the 'deferred' nodes."""
    for idx, other_extra in enumerate(self._others_extra):
        others = sorted(other_extra.values(), key=lambda x: self._path_to_key(x[0]))
        for other_path, other_ie in others:
            file_id = other_ie.file_id
            other_extra.pop(file_id)
            other_values = [(None, None)] * idx
            other_values.append((other_path, other_ie))
            for alt_idx, alt_extra in enumerate(self._others_extra[idx + 1:]):
                alt_idx = alt_idx + idx + 1
                alt_extra = self._others_extra[alt_idx]
                alt_tree = self._other_trees[alt_idx]
                other_values.append(self._lookup_by_file_id(alt_extra, alt_tree, file_id))
            yield (other_path, file_id, None, other_values)