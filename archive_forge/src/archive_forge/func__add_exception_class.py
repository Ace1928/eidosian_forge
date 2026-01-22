import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_exception_class(self, section, shape):
    breadcrumb_section = section.add_new_section('breadcrumb')
    breadcrumb_section.style.ref(self._client_class_name, f'../../../{self._service_name}')
    breadcrumb_section.write(f' / Client / exceptions / {shape.name}')
    section.add_title_section(shape.name)
    class_section = section.add_new_section(shape.name)
    class_name = self._exception_class_name(shape)
    class_section.style.start_sphinx_py_class(class_name=class_name)
    self._add_top_level_documentation(class_section, shape)
    self._add_exception_catch_example(class_section, shape)
    self._add_response_attr(class_section, shape)
    class_section.style.end_sphinx_py_class()