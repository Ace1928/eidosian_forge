from . import static_tuple
def _parse_leaf_lines(data, key_length, ref_list_length):
    lines = data.split(b'\n')
    nodes = []
    as_st = static_tuple.StaticTuple.from_sequence
    stuple = static_tuple.StaticTuple
    for line in lines[1:]:
        if line == b'':
            return nodes
        elements = line.split(b'\x00', key_length)
        key = as_st(elements[:key_length]).intern()
        line = elements[-1]
        references, value = line.rsplit(b'\x00', 1)
        if ref_list_length:
            ref_lists = []
            for ref_string in references.split(b'\t'):
                ref_list = as_st([as_st(ref.split(b'\x00')).intern() for ref in ref_string.split(b'\r') if ref])
                ref_lists.append(ref_list)
            ref_lists = as_st(ref_lists)
            node_value = stuple(value, ref_lists)
        else:
            node_value = stuple(value, stuple())
        nodes.append((key, node_value))
    return nodes