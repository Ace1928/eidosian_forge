from .api import FancyValidator
def variable_encode(d, prepend='', result=None, add_repetitions=True, dict_char='.', list_char='-'):
    """Encode a nested structure into a flat dictionary."""
    if result is None:
        result = {}
    if isinstance(d, dict):
        for key, value in d.items():
            if key is None:
                name = prepend
            elif not prepend:
                name = key
            else:
                name = '%s%s%s' % (prepend, dict_char, key)
            variable_encode(value, name, result, add_repetitions, dict_char=dict_char, list_char=list_char)
    elif isinstance(d, list):
        for i, value in enumerate(d):
            variable_encode(value, '%s%s%i' % (prepend, list_char, i), result, add_repetitions, dict_char=dict_char, list_char=list_char)
        if add_repetitions:
            rep_name = '%s--repetitions' % prepend if prepend else '__repetitions__'
            result[rep_name] = str(len(d))
    else:
        result[prepend] = d
    return result