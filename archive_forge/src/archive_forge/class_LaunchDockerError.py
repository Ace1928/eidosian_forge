from wandb.errors import Error
class LaunchDockerError(Error):
    """Raised when Docker daemon is not running."""
    pass