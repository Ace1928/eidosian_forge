import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_context_params_list(self, section):
    section.style.new_line()
    sn = f'``{self._service_name}``'
    section.writeln(f'The available {sn} client context params are:')
    for param in self._context_params:
        section.style.new_line()
        name = f'``{xform_name(param.name)}``'
        section.write(f'* {name} ({param.type}) - {param.documentation}')