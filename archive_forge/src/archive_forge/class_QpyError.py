from qiskit.exceptions import QiskitError, QiskitWarning
class QpyError(QiskitError):
    """Errors raised by the qpy module."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)