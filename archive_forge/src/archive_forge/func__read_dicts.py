def _read_dicts(fn_list, keyatom):
    """Read multiple files into a list of residue dictionaries (PRIVATE)."""
    dict_list = []
    datalabel_list = []
    for fn in fn_list:
        peaklist = Peaklist(fn)
        dictionary = peaklist.residue_dict(keyatom)
        dict_list.append(dictionary)
        datalabel_list.append(peaklist.datalabels)
    return [dict_list, datalabel_list]