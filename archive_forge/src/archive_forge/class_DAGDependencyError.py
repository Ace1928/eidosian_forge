from qiskit.exceptions import QiskitError
class DAGDependencyError(QiskitError):
    """Base class for errors raised by the DAGDependency object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)