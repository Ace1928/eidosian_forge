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
class CompilerBase(object):
    """
    Stores and manages states for the compiler
    """

    def __init__(self, typingctx, targetctx, library, args, return_type, flags, locals):
        config.reload_config()
        typingctx.refresh()
        targetctx.refresh()
        self.state = StateDict()
        self.state.typingctx = typingctx
        self.state.targetctx = _make_subtarget(targetctx, flags)
        self.state.library = library
        self.state.args = args
        self.state.return_type = return_type
        self.state.flags = flags
        self.state.locals = locals
        self.state.bc = None
        self.state.func_id = None
        self.state.func_ir = None
        self.state.lifted = None
        self.state.lifted_from = None
        self.state.typemap = None
        self.state.calltypes = None
        self.state.type_annotation = None
        self.state.metadata = {}
        self.state.reload_init = []
        self.state.pipeline = self
        self.state.parfor_diagnostics = ParforDiagnostics()
        self.state.metadata['parfor_diagnostics'] = self.state.parfor_diagnostics
        self.state.metadata['parfors'] = {}
        self.state.status = _CompileStatus(can_fallback=self.state.flags.enable_pyobject)

    def compile_extra(self, func):
        self.state.func_id = bytecode.FunctionIdentity.from_function(func)
        ExtractByteCode().run_pass(self.state)
        self.state.lifted = ()
        self.state.lifted_from = None
        return self._compile_bytecode()

    def compile_ir(self, func_ir, lifted=(), lifted_from=None):
        self.state.func_id = func_ir.func_id
        self.state.lifted = lifted
        self.state.lifted_from = lifted_from
        self.state.func_ir = func_ir
        self.state.nargs = self.state.func_ir.arg_count
        FixupArgs().run_pass(self.state)
        return self._compile_ir()

    def define_pipelines(self):
        """Child classes override this to customize the pipelines in use.
        """
        raise NotImplementedError()

    def _compile_core(self):
        """
        Populate and run compiler pipeline
        """
        with ConfigStack().enter(self.state.flags.copy()):
            pms = self.define_pipelines()
            for pm in pms:
                pipeline_name = pm.pipeline_name
                func_name = '%s.%s' % (self.state.func_id.modname, self.state.func_id.func_qualname)
                event('Pipeline: %s for %s' % (pipeline_name, func_name))
                self.state.metadata['pipeline_times'] = {pipeline_name: pm.exec_times}
                is_final_pipeline = pm == pms[-1]
                res = None
                try:
                    pm.run(self.state)
                    if self.state.cr is not None:
                        break
                except _EarlyPipelineCompletion as e:
                    res = e.result
                    break
                except Exception as e:
                    if utils.use_new_style_errors() and (not isinstance(e, errors.NumbaError)):
                        raise e
                    self.state.status.fail_reason = e
                    if is_final_pipeline:
                        raise e
            else:
                raise CompilerError('All available pipelines exhausted')
            self.state.pipeline = None
            if res is not None:
                return res
            else:
                assert self.state.cr is not None
                return self.state.cr

    def _compile_bytecode(self):
        """
        Populate and run pipeline for bytecode input
        """
        assert self.state.func_ir is None
        return self._compile_core()

    def _compile_ir(self):
        """
        Populate and run pipeline for IR input
        """
        assert self.state.func_ir is not None
        return self._compile_core()