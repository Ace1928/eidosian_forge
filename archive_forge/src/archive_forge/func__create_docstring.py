from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.paginator import document_paginate_method
from botocore.docs.waiter import document_wait_method
def _create_docstring(self):
    docstring_structure = DocumentStructure('docstring', target='html')
    self._write_docstring(docstring_structure, *self._gen_args, **self._gen_kwargs)
    return docstring_structure.flush_structure().decode('utf-8')