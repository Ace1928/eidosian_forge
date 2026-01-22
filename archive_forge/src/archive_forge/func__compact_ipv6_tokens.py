from struct import unpack as _unpack, pack as _pack
def _compact_ipv6_tokens(tokens):
    new_tokens = []
    positions = []
    start_index = None
    num_tokens = 0
    for idx, token in enumerate(tokens):
        if token == '0':
            if start_index is None:
                start_index = idx
            num_tokens += 1
        else:
            if num_tokens > 1:
                positions.append((num_tokens, start_index))
            start_index = None
            num_tokens = 0
        new_tokens.append(token)
    if num_tokens > 1:
        positions.append((num_tokens, start_index))
    if len(positions) != 0:
        positions.sort(key=lambda x: x[1])
        best_position = positions[0]
        for position in positions:
            if position[0] > best_position[0]:
                best_position = position
        length, start_idx = best_position
        new_tokens = new_tokens[0:start_idx] + [''] + new_tokens[start_idx + length:]
        if new_tokens[0] == '':
            new_tokens.insert(0, '')
        if new_tokens[-1] == '':
            new_tokens.append('')
    return new_tokens