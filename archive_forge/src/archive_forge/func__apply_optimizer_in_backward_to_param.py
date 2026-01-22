from typing import Any, Dict, Iterable, List, no_type_check, Type
import torch
@no_type_check
def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
    if param not in param_to_acc_grad_map:
        param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[0][0]
    optimizer = optimizer_class([param], **optimizer_kwargs)
    if not hasattr(param, '_in_backward_optimizers'):
        param._in_backward_optimizers = []
        param._optimizer_classes = []
        param._optimizer_kwargs = []
    param._in_backward_optimizers.append(optimizer)
    param._optimizer_classes.append(optimizer_class)
    param._optimizer_kwargs.append(optimizer_kwargs)
    if not register_hook:
        return

    def optimizer_hook(*_unused) -> None:
        for opt in param._in_backward_optimizers:
            opt.step()
        param.grad = None
    handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)
    if param not in param_to_optim_hook_handle_map:
        param_to_optim_hook_handle_map[param] = []
    param_to_optim_hook_handle_map[param].append(handle)