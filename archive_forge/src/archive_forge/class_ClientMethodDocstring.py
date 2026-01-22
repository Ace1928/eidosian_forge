from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.paginator import document_paginate_method
from botocore.docs.waiter import document_wait_method
class ClientMethodDocstring(LazyLoadedDocstring):

    def _write_docstring(self, *args, **kwargs):
        document_model_driven_method(*args, **kwargs)