import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_method_exceptions_list(self, section, operation_model):
    error_section = section.add_new_section('exceptions')
    error_section.style.new_line()
    error_section.style.bold('Exceptions')
    error_section.style.new_line()
    for error in operation_model.error_shapes:
        class_name = f'{self._client_class_name}.Client.exceptions.{error.name}'
        error_section.style.li(':py:class:`%s`' % class_name)