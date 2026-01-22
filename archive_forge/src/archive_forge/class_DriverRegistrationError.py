from click import FileError
class DriverRegistrationError(ValueError):
    """Raised when a format driver is requested but is not registered."""