def _refine_cherrypick_conflict(self, zstart, zend, astart, aend, bstart, bend):
    """When cherrypicking b => a, ignore matches with b and base."""
    matcher = self.sequence_matcher(None, self.base[zstart:zend], self.b[bstart:bend])
    matches = matcher.get_matching_blocks()
    last_base_idx = 0
    last_b_idx = 0
    last_b_idx = 0
    yielded_a = False
    for base_idx, b_idx, match_len in matches:
        conflict_b_len = b_idx - last_b_idx
        if conflict_b_len == 0:
            pass
        elif yielded_a:
            yield ('conflict', zstart + last_base_idx, zstart + base_idx, aend, aend, bstart + last_b_idx, bstart + b_idx)
        else:
            yielded_a = True
            yield ('conflict', zstart + last_base_idx, zstart + base_idx, astart, aend, bstart + last_b_idx, bstart + b_idx)
        last_base_idx = base_idx + match_len
        last_b_idx = b_idx + match_len
    if last_base_idx != zend - zstart or last_b_idx != bend - bstart:
        if yielded_a:
            yield ('conflict', zstart + last_base_idx, zstart + base_idx, aend, aend, bstart + last_b_idx, bstart + b_idx)
        else:
            yielded_a = True
            yield ('conflict', zstart + last_base_idx, zstart + base_idx, astart, aend, bstart + last_b_idx, bstart + b_idx)
    if not yielded_a:
        yield ('conflict', zstart, zend, astart, aend, bstart, bend)