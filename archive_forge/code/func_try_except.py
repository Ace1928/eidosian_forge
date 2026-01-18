import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
def try_except(self, code_options, cleanup: List[Instruction]):
    """
        Codegen based off of:
        load args
        enter context
        try:
            (rest)
        finally:
            exit context
        """
    load_args = []
    if self.target_values:
        load_args = [create_instruction('LOAD_CONST', argval=val) for val in self.target_values]
    ctx_name = unique_id(f'___context_manager_{self.stack_index}')
    if ctx_name not in code_options['co_varnames']:
        code_options['co_varnames'] += (ctx_name,)
    for name in ['__enter__', '__exit__']:
        if name not in code_options['co_names']:
            code_options['co_names'] += (name,)
    except_jump_target = create_instruction('NOP' if sys.version_info < (3, 11) else 'PUSH_EXC_INFO')
    cleanup_complete_jump_target = create_instruction('NOP')
    setup_finally = [*load_args, *create_call_function(len(load_args), True), create_instruction('STORE_FAST', argval=ctx_name), create_instruction('LOAD_FAST', argval=ctx_name), create_instruction('LOAD_METHOD', argval='__enter__'), *create_call_method(0), create_instruction('POP_TOP')]
    if sys.version_info < (3, 11):
        setup_finally.append(create_instruction('SETUP_FINALLY', target=except_jump_target))
    else:
        exn_tab_begin = create_instruction('NOP')
        exn_tab_end = create_instruction('NOP')
        exn_tab_begin.exn_tab_entry = InstructionExnTabEntry(exn_tab_begin, exn_tab_end, except_jump_target, self.stack_index + 1, False)
        setup_finally.append(exn_tab_begin)

    def create_reset():
        return [create_instruction('LOAD_FAST', argval=ctx_name), create_instruction('LOAD_METHOD', argval='__exit__'), create_instruction('LOAD_CONST', argval=None), create_dup_top(), create_dup_top(), *create_call_method(3), create_instruction('POP_TOP')]
    if sys.version_info < (3, 9):
        epilogue = [create_instruction('POP_BLOCK'), create_instruction('BEGIN_FINALLY'), except_jump_target, *create_reset(), create_instruction('END_FINALLY')]
    elif sys.version_info < (3, 11):
        epilogue = [create_instruction('POP_BLOCK'), *create_reset(), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), except_jump_target, *create_reset(), create_instruction('RERAISE'), cleanup_complete_jump_target]
    else:
        finally_exn_tab_end = create_instruction('RERAISE', arg=0)
        finally_exn_tab_target = create_instruction('COPY', arg=3)
        except_jump_target.exn_tab_entry = InstructionExnTabEntry(except_jump_target, finally_exn_tab_end, finally_exn_tab_target, self.stack_index + 2, True)
        epilogue = [exn_tab_end, *create_reset(), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), except_jump_target, *create_reset(), finally_exn_tab_end, finally_exn_tab_target, create_instruction('POP_EXCEPT'), create_instruction('RERAISE', arg=1), cleanup_complete_jump_target]
    cleanup[:] = epilogue + cleanup
    return setup_finally