import torch
from torch._C import _disabled_torch_function_impl
from collections import OrderedDict
class UninitializedTensorMixin:
    _allowed_methods = [torch.Tensor.__hash__, torch.Tensor.size, torch.Tensor.copy_, torch.Tensor.is_floating_point, torch.Tensor.half, torch.Tensor.float, torch.Tensor.double, torch.Tensor.char, torch.Tensor.short, torch.Tensor.int, torch.Tensor.long, torch.Tensor.cuda, torch.Tensor.cpu, torch.Tensor.to, torch.Tensor.get_device, torch._has_compatible_shallow_copy_type]

    def materialize(self, shape, device=None, dtype=None):
        """Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
        if device is None:
            device = self.data.device
        if dtype is None:
            dtype = self.data.dtype
        self.data = torch.empty(shape, device=device, dtype=dtype)
        self.__class__ = self.cls_to_become

    @property
    def shape(self):
        raise RuntimeError("Can't access the shape of an uninitialized parameter or buffer. This error usually happens in `load_state_dict` when trying to load an uninitialized parameter into an initialized one. Call `forward` to initialize the parameters before accessing their attributes.")

    def share_memory_(self):
        raise RuntimeError("Can't share memory on an uninitialized parameter or buffer. Call `forward` to initialize the parameters before calling `module.share_memory()`.")

    def __repr__(self):
        return f'<{self.__class__.__name__}>'

    def __reduce_ex__(self, proto):
        return (self.__class__, (self.requires_grad,))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in cls._allowed_methods or func.__class__.__name__ == 'method-wrapper':
            if kwargs is None:
                kwargs = {}
            return super().__torch_function__(func, types, args, kwargs)
        raise ValueError(f'Attempted to use an uninitialized parameter in {func}. This error happens when you are using a `LazyModule` or explicitly manipulating `torch.nn.parameter.{cls.__name__}` objects. When using LazyModules Call `forward` with a dummy batch to initialize the parameters before calling torch functions')