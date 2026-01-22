import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
class QueuingManager:
    """Singleton global entry point for managing active recording contexts.

    This class consists purely of class methods. It both maintains a list of
    recording queues and allows communication with the currently active object.

    Queueable objects, like :class:`~.operation.Operator` and :class:`~.measurements.MeasurementProcess`, should
    use ``QueuingManager`` as an entry point for accessing the active queue.

    See also: :class:`~.AnnotatedQueue`, :class:`~.tape.QuantumTape`, :meth:`~.operation.Operator.queue`.

    Recording queues, such as :class:`~.AnnotatedQueue`, must define the following methods:

    * ``append``: define an action to perform when an object append
      request is made.

    * ``remove``: define an action to perform when an object removal request is made.

    * ``get_info``: retrieve the object's metadata

    * ``update_info``: Update an object's metadata if it is already queued.

    To start and end recording, the recording queue can use the :meth:`add_active_queue` and
    :meth:`remove_active_queue` methods.

    """
    _active_contexts = []
    'The stack of contexts that are currently active.'

    @classmethod
    def add_active_queue(cls, queue):
        """Makes a queue the currently active recording context."""
        cls._active_contexts.append(queue)

    @classmethod
    def remove_active_queue(cls):
        """Ends recording on the currently active recording queue."""
        return cls._active_contexts.pop()

    @classmethod
    def recording(cls):
        """Whether a queuing context is active and recording operations"""
        return bool(cls._active_contexts)

    @classmethod
    def active_context(cls) -> Optional['AnnotatedQueue']:
        """Returns the currently active queuing context."""
        return cls._active_contexts[-1] if cls.recording() else None

    @classmethod
    @contextmanager
    def stop_recording(cls):
        """A context manager and decorator to ensure that contained logic is non-recordable
        or non-queueable within a QNode or quantum tape context.

        **Example:**

        Consider the function:

        >>> def list_of_ops(params, wires):
        ...     return [
        ...         qml.RX(params[0], wires=wires),
        ...         qml.RY(params[1], wires=wires),
        ...         qml.RZ(params[2], wires=wires)
        ...     ]

        If executed in a recording context, the operations constructed in the function will be queued:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     ops = list_of_ops(params, wires=0)
        ...     qml.apply(ops[-1])  # apply the last operation from the list again
        ...     return qml.expval(qml.Z(0))
        >>> print(qml.draw(circuit)([1, 2, 3]))
        0: ──RX(1.00)──RY(2.00)──RZ(3.00)──RZ(3.00)─┤  <Z>

        Using the ``stop_recording`` context manager, all logic contained inside is not queued or recorded.

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     with qml.QueuingManager.stop_recording():
        ...         ops = list_of_ops(params, wires=0)
        ...     qml.apply(ops[-1])
        ...     return qml.expval(qml.Z(0))
        >>> print(qml.draw(circuit)([1, 2, 3]))
        0: ──RZ(3.00)─┤  <Z>

        The context manager can also be used as a decorator on a function:

        >>> @qml.QueuingManager.stop_recording()
        ... def list_of_ops(params, wires):
        ...     return [
        ...         qml.RX(params[0], wires=wires),
        ...         qml.RY(params[1], wires=wires),
        ...         qml.RZ(params[2], wires=wires)
        ...     ]
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     ops = list_of_ops(params, wires=0)
        ...     qml.apply(ops[-1])
        ...     return qml.expval(qml.Z(0))
        >>> print(qml.draw(circuit)([1, 2, 3]))
        0: ──RZ(3.00)─┤  <Z>

        """
        previously_active_contexts = cls._active_contexts
        cls._active_contexts = []
        try:
            yield
        finally:
            cls._active_contexts = previously_active_contexts

    @classmethod
    def append(cls, obj, **kwargs):
        """Append an object to the queue(s).

        Args:
            obj: the object to be appended
        """
        if cls.recording():
            cls.active_context().append(obj, **kwargs)

    @classmethod
    def remove(cls, obj):
        """Remove an object from the queue(s) if it is in the queue(s).

        Args:
            obj: the object to be removed
        """
        if cls.recording():
            cls.active_context().remove(obj)

    @classmethod
    def update_info(cls, obj, **kwargs):
        """Updates information of an object in the active queue if it is already in the queue.

        Args:
            obj: the object with metadata to be updated
        """
        if cls.recording():
            cls.active_context().update_info(obj, **kwargs)

    @classmethod
    def get_info(cls, obj):
        """Retrieves information of an object in the active queue.

        Args:
            obj: the object with metadata to be retrieved

        Returns:
            object metadata
        """
        return cls.active_context().get_info(obj) if cls.recording() else None