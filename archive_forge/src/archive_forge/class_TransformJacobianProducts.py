import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
class TransformJacobianProducts(JacobianProductCalculator):
    """Compute VJPs, JVPs and Jacobians via a gradient transform :class:`~.TransformDispatcher`.

    Args:
        inner_execute (Callable[[Tuple[QuantumTape]], ResultBatch]): a function that
            executes the batch of circuits and returns their results.
        gradient_transform (.TransformDispatcher): the gradient transform to use.
        gradient_kwargs (dict): Any keyword arguments for the gradient transform.

    Keyword Args:
        cache_full_jacobian=False (bool): Whether or not to compute the full jacobian and cache it,
            instead of treating each call as independent. This keyword argument is used to patch problematic
            autograd behavior when caching is turned off. In this case, caching will be based on the identity
            of the batch, rather than the potentially expensive :attr:`~.QuantumScript.hash` that is used
            by :func:`~.cache_execute`.

    >>> inner_execute = qml.device('default.qubit').execute
    >>> gradient_transform = qml.gradients.param_shift
    >>> kwargs = {"broadcast": True}
    >>> jpc = TransformJacobianProducts(inner_execute, gradient_transform, kwargs)

    """

    def __repr__(self):
        return f'TransformJacobianProducts({self._inner_execute}, gradient_transform={self._gradient_transform}, gradient_kwargs={self._gradient_kwargs}, cache_full_jacobian={self._cache_full_jacobian})'

    def __init__(self, inner_execute: Callable, gradient_transform: 'pennylane.transforms.core.TransformDispatcher', gradient_kwargs: Optional[dict]=None, cache_full_jacobian: bool=False):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('TransformJacobianProduct being created with (%s, %s, %s, %s)', inspect.getsource(inner_execute) if logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(inner_execute) else inner_execute, gradient_transform, gradient_kwargs, cache_full_jacobian)
        self._inner_execute = inner_execute
        self._gradient_transform = gradient_transform
        self._gradient_kwargs = gradient_kwargs or {}
        self._cache_full_jacobian = cache_full_jacobian
        self._cache = LRUCache(maxsize=10)

    def execute_and_compute_jvp(self, tapes: Batch, tangents: Tuple[Tuple[TensorLike]]):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('execute_and_compute_jvp called with (%s, %s)', tapes, tangents)
        num_result_tapes = len(tapes)
        if self._cache_full_jacobian:
            jacs = self.compute_jacobian(tapes)
            jvps = _compute_jvps(jacs, tangents, tapes)
            return (self._inner_execute(tapes), jvps)
        jvp_tapes, jvp_processing_fn = qml.gradients.batch_jvp(tapes, tangents, self._gradient_transform, gradient_kwargs=self._gradient_kwargs)
        full_batch = tapes + tuple(jvp_tapes)
        full_results = self._inner_execute(full_batch)
        results = full_results[:num_result_tapes]
        jvp_results = full_results[num_result_tapes:]
        jvps = jvp_processing_fn(jvp_results)
        return (tuple(results), tuple(jvps))

    def compute_vjp(self, tapes: Batch, dy: Tuple[Tuple[TensorLike]]):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('compute_vjp called with (%s, %s)', tapes, dy)
        if self._cache_full_jacobian:
            jacs = self.compute_jacobian(tapes)
            return _compute_vjps(jacs, dy, tapes)
        vjp_tapes, processing_fn = qml.gradients.batch_vjp(tapes, dy, self._gradient_transform, gradient_kwargs=self._gradient_kwargs)
        vjp_results = self._inner_execute(tuple(vjp_tapes))
        return tuple(processing_fn(vjp_results))

    def execute_and_compute_jacobian(self, tapes: Batch):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('execute_and_compute_jacobian called with %s', tapes)
        num_result_tapes = len(tapes)
        jac_tapes, jac_postprocessing = self._gradient_transform(tapes, **self._gradient_kwargs)
        full_batch = tapes + tuple(jac_tapes)
        full_results = self._inner_execute(full_batch)
        results = full_results[:num_result_tapes]
        jac_results = full_results[num_result_tapes:]
        jacs = jac_postprocessing(jac_results)
        return (tuple(results), tuple(jacs))

    def compute_jacobian(self, tapes: Batch):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('compute_jacobian called with %s', tapes)
        if tapes in self._cache:
            return self._cache[tapes]
        jac_tapes, batch_post_processing = self._gradient_transform(tapes, **self._gradient_kwargs)
        results = self._inner_execute(jac_tapes)
        jacs = tuple(batch_post_processing(results))
        self._cache[tapes] = jacs
        return jacs