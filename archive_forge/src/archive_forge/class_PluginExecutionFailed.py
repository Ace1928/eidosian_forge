from __future__ import annotations
class PluginExecutionFailed(Flake8Exception):
    """The plugin failed during execution."""
    FORMAT = '{fname}: "{plugin}" failed during execution due to {exc!r}'

    def __init__(self, filename: str, plugin_name: str, exception: Exception) -> None:
        """Utilize keyword arguments for message generation."""
        self.filename = filename
        self.plugin_name = plugin_name
        self.original_exception = exception
        super().__init__(filename, plugin_name, exception)

    def __str__(self) -> str:
        """Format our exception message."""
        return self.FORMAT.format(fname=self.filename, plugin=self.plugin_name, exc=self.original_exception)