import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _filter_client_method(self, **kwargs):
    for filter in self._CLIENT_METHODS_FILTERS:
        filter_include = filter(**kwargs)
        if filter_include is not None:
            return filter_include
    return True