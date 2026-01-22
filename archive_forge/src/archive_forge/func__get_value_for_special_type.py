from botocore.utils import is_json_value_header
def _get_value_for_special_type(self, shape, special_type_map):
    if is_json_value_header(shape):
        return special_type_map['jsonvalue_header']
    if hasattr(shape, 'is_document_type') and shape.is_document_type:
        return special_type_map['document_type']
    for special_type, marked_shape in self._context['special_shape_types'].items():
        if special_type in special_type_map:
            if shape == marked_shape:
                return special_type_map[special_type]
    return None