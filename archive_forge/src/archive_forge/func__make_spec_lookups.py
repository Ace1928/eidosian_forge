def _make_spec_lookups(specs):
    lookup = {}
    by_status = {}
    by_type = {}
    for spec in specs:
        type_ = spec['type']
        status_byte = spec['status_byte']
        by_type[type_] = spec
        if status_byte in CHANNEL_MESSAGES:
            for channel in range(16):
                by_status[status_byte | channel] = spec
        else:
            by_status[status_byte] = spec
    lookup.update(by_status)
    lookup.update(by_type)
    return (lookup, by_status, by_type)