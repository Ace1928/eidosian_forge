import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
class DeviceDerivatives(JacobianProductCalculator):
    """Calculate jacobian products via a device provided jacobian.  This class relies on either ``qml.Device.gradients`` or
    ``qml.devices.Device.compute_derivatives``.

    Args:

        device (Union[pennylane.Device, pennylane.devices.Device]): the device for execution and derivatives.
            Must support first order gradients with the requested configuration.
        execution_config (pennylane.devices.ExecutionConfig): a datastructure containing the options needed to fully
           describe the execution. Only used with :class:`pennylane.devices.Device` from the new device interface.
        gradient_kwargs (dict): a dictionary of keyword arguments for the gradients. Only used with a :class:`~.pennylane.Device`
            from the old device interface.

    **Examples:**

    >>> device = qml.device('default.qubit')
    >>> config = qml.devices.ExecutionConfig(gradient_method="adjoint")
    >>> jpc = DeviceDerivatives(device, config, {})

    This same class can also be used with the old device interface.

    >>> device = qml.device('lightning.qubit', wires=5)
    >>> gradient_kwargs = {"method": "adjoint_jacobian"}
    >>> jpc_lightning = DeviceDerivatives(device, gradient_kwargs=gradient_kwargs)

    **Technical comments on caching and calculating the gradients on execution:**

    In order to store results and Jacobians for the backward pass during the forward pass,
    the ``_jacs_cache`` and ``_results_cache`` properties are ``LRUCache`` objects with a maximum size of 10.
    In the current execution pipeline, only one batch will be used per instance, but a size of 10 adds some extra
    flexibility for future uses.

    Note that batches of identically looking :class:`~.QuantumScript` s that are different instances will be cached separately.
    This is because the ``hash`` of  :class:`~.QuantumScript` is expensive, as it requires inspecting all its constituents,
    which is not worth the effort in this case.

    When a forward pass with :meth:`~.execute_and_cache_jacobian` is called, both the results and the jacobian for the object are stored.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.Z(0))])
    >>> batch = (tape, )
    >>> with device.tracker:
    ...     results = jpc.execute_and_cache_jacobian(batch )
    >>> results
    (0.5403023058681398,)
    >>> device.tracker.totals
    {'execute_and_derivative_batches': 1, 'executions': 1, 'derivatives': 1}
    >>> jpc._jacs_cache
    LRUCache({5660934048: (array(-0.84147098),)}, maxsize=10, currsize=1)

    Then when the vjp, jvp, or jacobian is requested, that cached value is used instead of requesting from
    the device again.

    >>> with device.tracker:
    ...     vjp = jpc.compute_vjp(batch , (0.5, ) )
    >>> vjp
    (array([-0.42073549]),)
    >>> device.tracker.totals
    {}

    """

    def __repr__(self):
        return f'<DeviceDerivatives: {self._device.name}, {self._gradient_kwargs}, {self._execution_config}>'

    def __init__(self, device: Union['qml.devices.Device', 'qml.Device'], execution_config: Optional['qml.devices.ExecutionConfig']=None, gradient_kwargs: dict=None):
        if gradient_kwargs is None:
            gradient_kwargs = {}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('DeviceDerivatives created with (%s, %s, %s)', device, execution_config, gradient_kwargs)
        self._device = device
        self._execution_config = execution_config
        self._gradient_kwargs = gradient_kwargs
        self._uses_new_device = not isinstance(device, qml.Device)
        self._results_cache = LRUCache(maxsize=10)
        self._jacs_cache = LRUCache(maxsize=10)

    def _dev_execute_and_compute_derivatives(self, tapes: Batch):
        """
        Converts tapes to numpy before computing the the results and derivatives on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in tapes))
        if self._uses_new_device:
            return self._device.execute_and_compute_derivatives(numpy_tapes, self._execution_config)
        return self._device.execute_and_gradients(numpy_tapes, **self._gradient_kwargs)

    def _dev_execute(self, tapes: Batch):
        """
        Converts tapes to numpy before computing just the results on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in tapes))
        if self._uses_new_device:
            return self._device.execute(numpy_tapes, self._execution_config)
        return self._device.batch_execute(numpy_tapes)

    def _dev_compute_derivatives(self, tapes: Batch):
        """
        Converts tapes to numpy before computing the derivatives on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in tapes))
        if self._uses_new_device:
            return self._device.compute_derivatives(numpy_tapes, self._execution_config)
        return self._device.gradients(numpy_tapes, **self._gradient_kwargs)

    def execute_and_cache_jacobian(self, tapes: Batch):
        """Forward pass used to cache the results and jacobians.

        Args:
            tapes (tuple[`~.QuantumScript`]): the batch of tapes to execute and take derivatives of

        Returns:
            ResultBatch: the results of the execution.

        Side Effects:
            Caches both the results and jacobian into ``_results_cache`` and ``_jacs_cache``.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Forward pass called with %s', tapes)
        results, jac = self._dev_execute_and_compute_derivatives(tapes)
        self._results_cache[tapes] = results
        self._jacs_cache[tapes] = jac
        return results

    def execute_and_compute_jvp(self, tapes: Batch, tangents):
        """Calculate both the results for a batch of tapes and the jvp.

        This method is required to compute JVPs in the JAX interface.

        Args:
            tapes (tuple[`~.QuantumScript`]): The batch of tapes to take the derivatives of
            tangents (Sequence[Sequence[TensorLike]]): the tangents for the parameters of the tape.
                The ``i`` th tangent corresponds to the ``i`` th tape, and the ``j`` th entry into a
                tangent entry corresponds to the ``j`` th trainable parameter of the tape.

        Returns:
            ResultBatch, TensorLike: the results of the execution and the jacobian vector product

        Side Effects:
            caches newly computed results or jacobians if they were not already cached.

        **Examples:**

        For an instance of :class:`~.DeviceDerivatives` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.Z(0))])
        >>> batch = (tape0, tape1)
        >>> tangents0 = (1.5, )
        >>> tangents1 = (2.0, )
        >>> tangents = (tangents0, tangents1)
        >>> results, jvps = jpc.execute_and_compute_jvp(batch, tangents)
        >>> expected_results = (np.cos(0.1), np.cos(0.2))
        >>> qml.math.allclose(results, expected_results)
        True
        >>> jvps
        (array(-0.14975012), array(-0.39733866))
        >>> expected_jvps = 1.5 * -np.sin(0.1), 2.0 * -np.sin(0.2)
        >>> qml.math.allclose(jvps, expected_jvps)
        True

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """
        results, jacs = self.execute_and_compute_jacobian(tapes)
        jvps = _compute_jvps(jacs, tangents, tapes)
        return (results, jvps)

    def compute_vjp(self, tapes, dy):
        """Compute the vjp for a given batch of tapes.

        This method is used by autograd, torch, and tensorflow to compute VJPs.

        Args:
            tapes (tuple[`~.QuantumScript`]): the batch of tapes to take the derivatives of
            dy (tuple[tuple[TensorLike]]): the derivatives of the results of an execution.
                The ``i`` th entry (cotangent) corresponds to the ``i`` th tape, and the ``j`` th entry of the ``i`` th
                cotangent corresponds to the ``j`` th return value of the ``i`` th tape.

        Returns:
            TensorLike: the vector jacobian product.

        Side Effects:
            caches the newly computed jacobian if it wasn't already present in the cache.

        **Examples:**

        For an instance of :class:`~.DeviceDerivatives` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.Z(0)), qml.expval(qml.X(0))])
        >>> batch = (tape0, tape1)
        >>> dy0 = (0.5, )
        >>> dy1 = (2.0, 3.0)
        >>> dys = (dy0, dy1)
        >>> vjps = jpc.compute_vjp(batch, dys)
        >>> vjps
        (array([-0.04991671]), array([2.54286107]))
        >>> expected_vjp0 = 0.5 * -np.sin(0.1)
        >>> qml.math.allclose(vjps[0], expected_vjp0)
        True
        >>> expected_jvp1 = 2.0 * -np.sin(0.2) + 3.0 * np.cos(0.2)
        >>> qml.math.allclose(vjps[1], expected_vjp1)
        True

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """
        if tapes in self._jacs_cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(' %s : Retrieving jacobian from cache.', self)
            jacs = self._jacs_cache[tapes]
        else:
            jacs = self._dev_compute_derivatives(tapes)
            self._jacs_cache[tapes] = jacs
        return _compute_vjps(jacs, dy, tapes)

    def compute_jacobian(self, tapes):
        """Compute the full Jacobian for a batch of tapes.

        This method is required to compute Jacobians in the ``jax-jit`` interface

        Args:
            tapes: the batch of tapes to take the Jacobian of

        Returns:
            TensorLike: the full jacobian

        Side Effects:
            caches the newly computed jacobian if it wasn't already present in the cache.

        **Examples:**

        For an instance of :class:`~.DeviceDerivatives` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.Z(0)), qml.expval(qml.X(0))])
        >>> batch = (tape0, tape1)
        >>> jpc.compute_jacobian(batch)
        (array(-0.09983342), (array(-0.19866933), array(0.98006658)))

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """
        if tapes in self._jacs_cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('%s : Retrieving jacobian from cache.', self)
            return self._jacs_cache[tapes]
        jacs = self._dev_compute_derivatives(tapes)
        self._jacs_cache[tapes] = jacs
        return jacs

    def execute_and_compute_jacobian(self, tapes):
        if tapes not in self._results_cache and tapes not in self._jacs_cache:
            results, jacs = self._dev_execute_and_compute_derivatives(tapes)
            self._results_cache[tapes] = results
            self._jacs_cache[tapes] = jacs
            return (results, jacs)
        if tapes not in self._jacs_cache:
            raise NotImplementedError('No path to cache results without caching jac. This branch should not occur.')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('%s : Retrieving jacobian from cache.', self)
        jacs = self._jacs_cache[tapes]
        if tapes in self._results_cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('%s : Retrieving results from cache.', self)
            results = self._results_cache[tapes]
        else:
            results = self._dev_execute(tapes)
            self._results_cache[tapes] = results
        return (results, jacs)