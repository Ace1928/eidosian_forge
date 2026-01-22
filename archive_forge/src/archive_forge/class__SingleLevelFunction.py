import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch._C as _C
import torch._functorch as _functorch
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call
class _SingleLevelFunction(_C._FunctionBase, FunctionCtx, _HookMixin, metaclass=FunctionMeta):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """Define the forward of the custom autograd Function.

        This function is to be overridden by all subclasses.
        There are two ways to define forward:

        Usage 1 (Combined forward and ctx)::

            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass

        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details

        Usage 2 (Separate forward and ctx)::

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass

            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass

        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details

        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.
        """
        raise NotImplementedError('You must implement the forward function for custom autograd.Function.')

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
        """There are two ways to define the forward pass of an autograd.Function.

        Either:

        1. Override forward with the signature forward(ctx, *args, **kwargs).
           ``setup_context`` is not overridden. Setting up the ctx for backward
           happens inside the ``forward``.
        2. Override forward with the signature forward(*args, **kwargs) and
           override ``setup_context``. Setting up the ctx for backward happens
           inside ``setup_context`` (as opposed to inside the ``forward``)

        See :meth:`torch.autograd.Function.forward` and :ref:`extending-autograd` for more details.
        """
        raise NotImplementedError('setup_context is not implemented.')

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Define a formula for differentiating the operation with backward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.
        """
        raise NotImplementedError('You must implement either the backward or vjp method for your custom autograd.Function to use it with backward mode AD.')
    vjp = backward

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        """Define a formula for differentiating the operation with forward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many inputs as the :func:`forward` got (None will be passed in
        for non tensor inputs of the forward function),
        and it should return as many tensors as there were outputs to
        :func:`forward`. Each argument is the gradient w.r.t the given input,
        and each returned value should be the gradient w.r.t. the
        corresponding output. If an output is not a Tensor or the function is not
        differentiable with respect to that output, you can just pass None as a
        gradient for that input.

        You can use the :attr:`ctx` object to pass any value from the forward to this
        functions.
        """
        raise NotImplementedError('You must implement the jvp function for custom autograd.Function to use it with forward mode AD.')