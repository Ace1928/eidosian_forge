from qiskit.exceptions import QiskitError
class CalibrationNotAvailable(QiskitError):
    """Raised when calibration generation fails.

    .. note::
        This error is meant to caught by CalibrationBuilder and ignored.
    """