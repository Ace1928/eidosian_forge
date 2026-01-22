import timeit
from abc import abstractmethod, ABCMeta
from collections import namedtuple, OrderedDict
import inspect
from pprint import pformat
from numba.core.compiler_lock import global_compiler_lock
from numba.core import errors, config, transforms, utils
from numba.core.tracing import event
from numba.core.postproc import PostProcessor
from numba.core.ir_utils import enforce_no_dels, legalize_single_scope
import numba.core.event as ev
@global_compiler_lock
def _runPass(self, index, pss, internal_state):
    mutated = False

    def check(func, compiler_state):
        mangled = func(compiler_state)
        if mangled not in (True, False):
            msg = "CompilerPass implementations should return True/False. CompilerPass with name '%s' did not."
            raise ValueError(msg % pss.name())
        return mangled

    def debug_print(pass_name, print_condition, printable_condition):
        if pass_name in print_condition:
            fid = internal_state.func_id
            args = (fid.modname, fid.func_qualname, self.pipeline_name, printable_condition, pass_name)
            print(('%s.%s: %s: %s %s' % args).center(120, '-'))
            if internal_state.func_ir is not None:
                internal_state.func_ir.dump()
            else:
                print('func_ir is None')
    debug_print(pss.name(), self._print_before + self._print_wrap, 'BEFORE')
    pss.analysis = self._analysis
    qualname = internal_state.func_id.func_qualname
    ev_details = dict(name=f'{pss.name()} [{qualname}]', qualname=qualname, module=internal_state.func_id.modname, flags=pformat(internal_state.flags.values()), args=str(internal_state.args), return_type=str(internal_state.return_type))
    with ev.trigger_event('numba:run_pass', data=ev_details):
        with SimpleTimer() as init_time:
            mutated |= check(pss.run_initialization, internal_state)
        with SimpleTimer() as pass_time:
            mutated |= check(pss.run_pass, internal_state)
        with SimpleTimer() as finalize_time:
            mutated |= check(pss.run_finalizer, internal_state)
    if isinstance(pss, FunctionPass):
        enforce_no_dels(internal_state.func_ir)
    if self._ENFORCING:
        if _pass_registry.get(pss.__class__).mutates_CFG:
            if mutated:
                PostProcessor(internal_state.func_ir).run()
            else:
                internal_state.func_ir.blocks = transforms.canonicalize_cfg(internal_state.func_ir.blocks)
        if not legalize_single_scope(internal_state.func_ir.blocks):
            raise errors.CompilerError(f'multiple scope in func_ir detected in {pss}')
    pt = pass_timings(init_time.elapsed, pass_time.elapsed, finalize_time.elapsed)
    self.exec_times['%s_%s' % (index, pss.name())] = pt
    debug_print(pss.name(), self._print_after + self._print_wrap, 'AFTER')