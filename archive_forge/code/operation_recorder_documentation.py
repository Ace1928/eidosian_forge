from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from .tape import QuantumScript

        Overrides the default because OperationRecorder is both a QuantumScript and an AnnotatedQueue.

        If key is an int, the caller is likely indexing the backing QuantumScript. Otherwise, the
        caller is likely indexing the backing AnnotatedQueue.
        