import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_response_params(self, section, shape):
    params_section = section.add_new_section('Structure')
    params_section.style.new_line()
    params_section.style.bold('Structure')
    params_section.style.new_paragraph()
    documenter = ResponseParamsDocumenter(service_name=self._service_name, operation_name=None, event_emitter=self._client.meta.events)
    documenter.document_params(params_section, shape, include=[self._GENERIC_ERROR_SHAPE])