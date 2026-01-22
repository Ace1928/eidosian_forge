def _pair_set_add(data, a, b, are_mutually_exclusive):
    sub_dict = data.get(a)
    if not sub_dict:
        sub_dict = {}
        data[a] = sub_dict
    sub_dict[b] = are_mutually_exclusive