def _make_ambiguous_ranges(mydict, weight_table):
    range_d = {}
    avg_d = {}
    for letter, values in mydict.items():
        if len(values) == 1 and values[0] not in weight_table:
            continue
        weights = [weight_table.get(x) for x in values]
        range_d[letter] = (min(weights), max(weights))
        total_w = 0.0
        for w in weights:
            total_w = total_w + w
        avg_d[letter] = total_w / len(weights)
    return (range_d, avg_d)