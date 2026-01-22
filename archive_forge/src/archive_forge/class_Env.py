import inspect
import os
import sys
class Env(dict):
    """A dict for environment variables."""

    @staticmethod
    def snapshot():
        """Returns a snapshot of the current environment."""
        return Env(os.environ)

    def copy(self, updated_from=None):
        result = Env(self)
        if updated_from is not None:
            result.update(updated_from)
        return result

    def prepend_to(self, key, entry):
        """Prepends a new entry to a PATH-style environment variable, creating
        it if it doesn't exist already.
        """
        try:
            tail = os.path.pathsep + self[key]
        except KeyError:
            tail = ''
        self[key] = entry + tail