def update_class_to_generate_type(classes_to_generate, class_to_generate):
    properties = class_to_generate.get('properties')
    for _prop_name, prop_val in properties.items():
        prop_type = prop_val.get('type', '')
        if not prop_type:
            prop_type = prop_val.pop('$ref', '')
            if prop_type:
                assert prop_type.startswith('#/definitions/')
                prop_type = prop_type[len('#/definitions/'):]
                prop_val['type'] = Ref(prop_type, classes_to_generate[prop_type])