import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import Label, SetLineno, Instr
def _compute_stack_size(block, size, maxsize, *, check_pre_and_post=True):
    """Generator used to reduce the use of function stacks.

    This allows to avoid nested recursion and allow to treat more cases.

    HOW-TO:
        Following the methods of Trampoline
        (see https://en.wikipedia.org/wiki/Trampoline_(computing)),

        We yield either:

        - the arguments that would be used in the recursive calls, i.e,
          'yield block, size, maxsize' instead of making a recursive call
          '_compute_stack_size(block, size, maxsize)', if we encounter an
          instruction jumping to another block or if the block is linked to
          another one (ie `next_block` is set)
        - the required stack from the stack if we went through all the instructions
          or encountered an unconditional jump.

        In the first case, the calling function is then responsible for creating a
        new generator with those arguments, iterating over it till exhaustion to
        determine the stacksize required by the block and resuming this function
        with the determined stacksize.

    """
    if block.seen or block.startsize >= size:
        yield maxsize

    def update_size(pre_delta, post_delta, size, maxsize):
        size += pre_delta
        if size < 0:
            msg = 'Failed to compute stacksize, got negative size'
            raise RuntimeError(msg)
        size += post_delta
        maxsize = max(maxsize, size)
        return (size, maxsize)
    block.seen = True
    block.startsize = size
    for instr in block:
        if isinstance(instr, SetLineno):
            continue
        if instr.has_jump():
            effect = instr.pre_and_post_stack_effect(jump=True) if check_pre_and_post else (instr.stack_effect(jump=True), 0)
            taken_size, maxsize = update_size(*effect, size, maxsize)
            maxsize = (yield (instr.arg, taken_size, maxsize))
            if instr.is_uncond_jump():
                block.seen = False
                yield maxsize
        effect = instr.pre_and_post_stack_effect(jump=False) if check_pre_and_post else (instr.stack_effect(jump=False), 0)
        size, maxsize = update_size(*effect, size, maxsize)
    if block.next_block:
        maxsize = (yield (block.next_block, size, maxsize))
    block.seen = False
    yield maxsize