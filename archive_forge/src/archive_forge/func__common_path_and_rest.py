def _common_path_and_rest(l1, l2, common=[]):
    if len(l1) < 1:
        return (common, l1, l2)
    if len(l2) < 1:
        return (common, l1, l2)
    if l1[0] != l2[0]:
        return (common, l1, l2)
    return _common_path_and_rest(l1[1:], l2[1:], common + [l1[0:1]])