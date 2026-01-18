def update_class_to_generate_objects(classes_to_generate, class_to_generate):
    properties = class_to_generate['properties']
    for key, val in properties.items():
        if 'type' not in val:
            val['type'] = 'TypeNA'
            continue
        if val['type'] == 'object':
            create_new = val.copy()
            create_new.update({'name': '%s%s' % (class_to_generate['name'], key.title()), 'description': '    "%s" of %s' % (key, class_to_generate['name'])})
            if 'properties' not in create_new:
                create_new['properties'] = {}
            assert create_new['name'] not in classes_to_generate
            classes_to_generate[create_new['name']] = create_new
            update_class_to_generate_type(classes_to_generate, create_new)
            update_class_to_generate_props(create_new)
            update_class_to_generate_objects(classes_to_generate, create_new)
            val['type'] = Ref(create_new['name'], classes_to_generate[create_new['name']])
            val.pop('properties', None)