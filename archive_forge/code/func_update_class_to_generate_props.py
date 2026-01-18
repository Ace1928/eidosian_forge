def update_class_to_generate_props(class_to_generate):
    import json

    def default(o):
        if isinstance(o, Ref):
            return o.ref
        raise AssertionError('Unhandled: %s' % (o,))
    properties = class_to_generate['properties']
    class_to_generate['props'] = '    __props__ = %s' % _indent_lines(json.dumps(properties, indent=4, default=default)).strip()