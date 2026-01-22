import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from functorch.compile import compiled_function, min_cut_rematerialization_partition, nop
from .make_fx import randomize
import re
def _test_aot_autograd_forwards_backwards_helper(f, compiled_f, args, assert_raises_regex_fn, assert_equals_fn, try_check_data_specialization):

    def call_forwards_backwards(f, args):
        flat_args = pytree.arg_tree_leaves(*args)
        diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and arg.requires_grad]
        out = wrapper_set_seed(f, args)
        flat_out = pytree.tree_leaves(out)
        sm = 0
        for i in flat_out:
            if isinstance(i, torch.Tensor):
                sm += i.sum().abs()
        assert isinstance(sm, torch.Tensor)
        return (out, torch.autograd.grad(sm, diff_args, allow_unused=True))

    def check(args, ignore_failure=False):
        try:
            orig_out, orig_grad = call_forwards_backwards(f, args)
        except Exception:
            if ignore_failure:
                return
            raise
        if all((x is None for x in orig_grad)):
            with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
                call_forwards_backwards(compiled_f, args)
            return
        msg = "Gradients of the operator are different in eager-mode PyTorch vs AOTAutograd. This means the operator will have incorrect gradients underneath torch.compile. This could be because the operator's backward is incorrectly registered or not traceable or that there is a bug in AOTAutograd."
        compiled_out, compiled_grad = call_forwards_backwards(compiled_f, args)
        assert_equals_fn(compiled_out, orig_out, msg=outputs_msg)
        assert_equals_fn(compiled_grad, orig_grad, msg=msg)
    check(args, ignore_failure=False)
    if try_check_data_specialization:
        args = randomize(args)
        check(args, ignore_failure=True)