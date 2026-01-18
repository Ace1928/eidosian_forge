import logging
import os
import sys
import tempfile
from typing import Any, Dict
import torch
def throw_abstract_impl_not_imported_error(opname, module, context):
    if module in sys.modules:
        raise NotImplementedError(f'{opname}: We could not find the abstract impl for this operator. ')
    else:
        raise NotImplementedError(f"{opname}: We could not find the abstract impl for this operator. The operator specified that you may need to import the '{module}' Python module to load the abstract impl. {context}")