import sys
def _query_expansion(self, name, value, explode, prefix):
    """Expansion method for the '?' and '&' operators."""
    if value is None:
        return None
    tuples, items = is_list_of_tuples(value)
    safe = self.safe
    if list_test(value) and (not tuples):
        if not value:
            return None
        if explode:
            return self.join_str.join(('{}={}'.format(name, quote(v, safe)) for v in value))
        else:
            value = ','.join((quote(v, safe) for v in value))
            return '{}={}'.format(name, value)
    if dict_test(value) or tuples:
        if not value:
            return None
        items = items or sorted(value.items())
        if explode:
            return self.join_str.join(('{}={}'.format(quote(k, safe), quote(v, safe)) for k, v in items))
        else:
            value = ','.join(('{},{}'.format(quote(k, safe), quote(v, safe)) for k, v in items))
            return '{}={}'.format(name, value)
    if value:
        value = value[:prefix] if prefix else value
        return '{}={}'.format(name, quote(value, safe))
    return name + '='