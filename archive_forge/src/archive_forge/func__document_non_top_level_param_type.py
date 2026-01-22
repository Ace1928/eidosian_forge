from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def _document_non_top_level_param_type(self, type_section, shape):
    special_py_type = self._get_special_py_type_name(shape)
    py_type = py_type_name(shape.type_name)
    type_format = '(%s) --'
    if special_py_type is not None:
        type_section.write(type_format % special_py_type)
    else:
        type_section.style.italics(type_format % py_type)
    type_section.write(' ')