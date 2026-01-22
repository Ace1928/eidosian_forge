from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def _end_nested_param(self, section):
    section.style.dedent()
    section.style.new_line()