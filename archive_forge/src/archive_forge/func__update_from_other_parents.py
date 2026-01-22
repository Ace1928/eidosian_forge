from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _update_from_other_parents(self, key, annotations, lines, this_annotation, parent_key):
    """Reannotate this text relative to a second (or more) parent."""
    parent_annotations, matching_blocks = self._get_parent_annotations_and_matches(key, lines, parent_key)
    last_ann = None
    last_parent = None
    last_res = None
    for parent_idx, lines_idx, match_len in matching_blocks:
        ann_sub = annotations[lines_idx:lines_idx + match_len]
        par_sub = parent_annotations[parent_idx:parent_idx + match_len]
        if ann_sub == par_sub:
            continue
        for idx in range(match_len):
            ann = ann_sub[idx]
            par_ann = par_sub[idx]
            ann_idx = lines_idx + idx
            if ann == par_ann:
                continue
            if ann == this_annotation:
                annotations[ann_idx] = par_ann
                continue
            if ann == last_ann and par_ann == last_parent:
                annotations[ann_idx] = last_res
            else:
                new_ann = set(ann)
                new_ann.update(par_ann)
                new_ann = tuple(sorted(new_ann))
                annotations[ann_idx] = new_ann
                last_ann = ann
                last_parent = par_ann
                last_res = new_ann