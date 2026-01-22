class SSHExecCommandFailed(IntegrationException):
    """Raised when remotely executed command returns nonzero status."""
    message = "Command '%(command)s', exit status: %(exit_status)d, Error:\n%(strerror)s"