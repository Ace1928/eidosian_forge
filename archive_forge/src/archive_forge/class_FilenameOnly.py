from __future__ import annotations
from flake8.formatting import base
from flake8.violation import Violation
class FilenameOnly(SimpleFormatter):
    """Only print filenames, e.g., flake8 -q."""
    error_format = '%(path)s'

    def after_init(self) -> None:
        """Initialize our set of filenames."""
        self.filenames_already_printed: set[str] = set()

    def show_source(self, error: Violation) -> str | None:
        """Do not include the source code."""

    def format(self, error: Violation) -> str | None:
        """Ensure we only print each error once."""
        if error.filename not in self.filenames_already_printed:
            self.filenames_already_printed.add(error.filename)
            return super().format(error)
        else:
            return None