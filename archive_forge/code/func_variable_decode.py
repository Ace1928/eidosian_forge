from .api import FancyValidator
def variable_decode(d, dict_char='.', list_char='-'):
    """Decode the flat dictionary d into a nested structure."""
    result = {}
    dicts_to_sort = set()
    known_lengths = {}
    for key, value in d.items():
        keys = key.split(dict_char)
        new_keys = []
        was_repetition_count = False
        for key in keys:
            if key.endswith('--repetitions'):
                key = key[:-len('--repetitions')]
                new_keys.append(key)
                known_lengths[tuple(new_keys)] = int(value)
                was_repetition_count = True
                break
            elif list_char in key:
                maybe_key, index = key.split(list_char, 1)
                if not index.isdigit():
                    new_keys.append(key)
                else:
                    key = maybe_key
                    new_keys.append(key)
                    dicts_to_sort.add(tuple(new_keys))
                    new_keys.append(int(index))
            else:
                new_keys.append(key)
        if was_repetition_count:
            continue
        place = result
        for i in range(len(new_keys) - 1):
            try:
                if not isinstance(place[new_keys[i]], dict):
                    place[new_keys[i]] = {None: place[new_keys[i]]}
                place = place[new_keys[i]]
            except KeyError:
                place[new_keys[i]] = {}
                place = place[new_keys[i]]
        if new_keys[-1] in place:
            if isinstance(place[new_keys[-1]], dict):
                place[new_keys[-1]][None] = value
            elif isinstance(place[new_keys[-1]], list):
                if isinstance(value, list):
                    place[new_keys[-1]].extend(value)
                else:
                    place[new_keys[-1]].append(value)
            elif isinstance(value, list):
                place[new_keys[-1]] = [place[new_keys[-1]]]
                place[new_keys[-1]].extend(value)
            else:
                place[new_keys[-1]] = [place[new_keys[-1]], value]
        else:
            place[new_keys[-1]] = value
    to_sort_list = sorted(dicts_to_sort, key=len, reverse=True)
    for key in to_sort_list:
        to_sort = result
        source = None
        last_key = None
        for sub_key in key:
            source = to_sort
            last_key = sub_key
            to_sort = to_sort[sub_key]
        if None in to_sort:
            none_values = [(0, x) for x in to_sort.pop(None)]
            none_values.extend(to_sort.items())
            to_sort = none_values
        else:
            to_sort = to_sort.items()
        to_sort = [x[1] for x in sorted(to_sort, key=_sort_key)]
        if key in known_lengths:
            if len(to_sort) < known_lengths[key]:
                to_sort.extend([''] * (known_lengths[key] - len(to_sort)))
        source[last_key] = to_sort
    return result