import warnings
from autograd import jacobian as _jacobian
from autograd.core import make_vjp as _make_vjp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
def _get_grad_fn(self, args):
    """Get the required gradient function.

        * If the differentiable argnum was provided on initialization,
          this has been pre-computed and is available via self._grad_fn

        * Otherwise, we must dynamically construct the gradient function by
          inspecting as to which of the parameter arguments are marked
          as differentiable.
        """
    if self._grad_fn is not None:
        return (self._grad_fn, self._argnum)
    argnum = []
    for idx, arg in enumerate(args):
        trainable = getattr(arg, 'requires_grad', None) or isinstance(arg, ArrayBox)
        if trainable:
            if arg.dtype.name[:3] == 'int':
                raise ValueError('Autograd does not support differentiation of ints.')
            argnum.append(idx)
    if len(argnum) == 1:
        argnum = argnum[0]
    return (self._grad_with_forward(self._fun, argnum=argnum), argnum)