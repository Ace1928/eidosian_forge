from collections import namedtuple
import copy
import warnings
from numba.core.tracing import event
from numba.core import (utils, errors, interpreter, bytecode, postproc, config,
from numba.parfors.parfor import ParforDiagnostics
from numba.core.errors import CompilerError
from numba.core.environment import lookup_environment
from numba.core.compiler_machinery import PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.object_mode_passes import (ObjectModeFrontEnd,
from numba.core.targetconfig import TargetConfig, Option, ConfigStack
class Compiler(CompilerBase):
    """The default compiler
    """

    def define_pipelines(self):
        if self.state.flags.force_pyobject:
            return [DefaultPassBuilder.define_objectmode_pipeline(self.state)]
        else:
            return [DefaultPassBuilder.define_nopython_pipeline(self.state)]