from qiskit.exceptions import QiskitError
class PulseError(QiskitError):
    """Errors raised by the pulse module."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)