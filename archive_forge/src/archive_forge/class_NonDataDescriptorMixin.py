import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
from docutils.statemachine import StringList
import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint
class NonDataDescriptorMixin(DataDocumenterMixinBase):
    """
    Mixin for AttributeDocumenter to provide the feature for supporting non
    data-descriptors.

    .. note:: This mix-in must be inherited after other mix-ins.  Otherwise, docstring
              and :value: header will be suppressed unexpectedly.
    """

    def import_object(self, raiseerror: bool=False) -> bool:
        ret = super().import_object(raiseerror)
        if ret and (not inspect.isattributedescriptor(self.object)):
            self.non_data_descriptor = True
        else:
            self.non_data_descriptor = False
        return ret

    def should_suppress_value_header(self) -> bool:
        return not getattr(self, 'non_data_descriptor', False) or super().should_suppress_directive_header()

    def get_doc(self) -> Optional[List[List[str]]]:
        if getattr(self, 'non_data_descriptor', False):
            return None
        else:
            return super().get_doc()