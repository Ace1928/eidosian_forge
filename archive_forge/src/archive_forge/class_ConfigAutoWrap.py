import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast
import torch.nn as nn
class ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """
    in_autowrap_context: bool = False
    move_module_cuda_half: bool = False
    wrapper_cls: Optional[Callable] = None
    kwargs: Dict[str, Any] = {}
    auto_wrap_policy: Optional[Callable] = None

    def __init__(self, auto_wrap_policy: Optional[Callable]=None, **kwargs: Dict[str, Any]):
        self.auto_wrap_policy = auto_wrap_policy
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(auto_wrap_policy: Optional[Callable], kwargs: Any) -> None:
        if ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError('You are already within an autowrap context and we currently do not supported nested autowrap.')
        ConfigAutoWrap.in_autowrap_context = True
        if 'move_module_cuda_half' in kwargs.keys():
            ConfigAutoWrap.move_module_cuda_half = cast(bool, kwargs['move_module_cuda_half'])
            del kwargs['move_module_cuda_half']
        assert 'wrapper_cls' in kwargs.keys()
        ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs['wrapper_cls'])
        del kwargs['wrapper_cls']
        ConfigAutoWrap.auto_wrap_policy = default_auto_wrap_policy if auto_wrap_policy is None else auto_wrap_policy
        ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        ConfigAutoWrap.in_autowrap_context = False
        ConfigAutoWrap.move_module_cuda_half = False
        ConfigAutoWrap.wrapper_cls = None
        ConfigAutoWrap.kwargs = {}
        ConfigAutoWrap.auto_wrap_policy = None

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.auto_wrap_policy, self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()

    @staticmethod
    def recursive_wrap(module: nn.Module, auto_wrap_policy: Optional[Callable], module_is_root: bool, **kwargs: Any) -> Tuple[nn.Module, int]:
        """
        Automatically wrap child modules of *module* that meet the given
        criteria with :func:`auto_wrap`.

        Args:
            module (nn.Module):
                module to recursively wrap
            auto_wrap_policy (Callable, Optional):
                optionally, override the :func:`auto_wrap_policy` from the context.

        Returns:
            (nn.Module, int):
                Wrapped module and the number parameters wrapped recursively.
        """
        if auto_wrap_policy is None:
            auto_wrap_policy = ConfigAutoWrap.auto_wrap_policy
        for _, child in module.named_modules():
            assert not isinstance(child, cast(type, ConfigAutoWrap.wrapper_cls))
        num_params = sum([p.numel() for p in module.parameters()])
        assert auto_wrap_policy is not None
        if auto_wrap_policy(module=module, recurse=True, unwrapped_params=num_params, module_is_root=module_is_root):
            total_wrapped_params = 0
            for name, child in module.named_children():
                wrapped_child, num_wrapped_params = ConfigAutoWrap.recursive_wrap(module=child, auto_wrap_policy=auto_wrap_policy, module_is_root=False, **kwargs)
                setattr(module, name, wrapped_child)
                total_wrapped_params += num_wrapped_params
            remainder = num_params - total_wrapped_params
            if auto_wrap_policy(module=module, recurse=False, unwrapped_params=remainder, module_is_root=module_is_root):
                return (wrap(module, **kwargs), num_params)
            else:
                return (module, total_wrapped_params)
        return (module, 0)