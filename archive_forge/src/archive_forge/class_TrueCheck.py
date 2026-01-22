import abc
import ast
import inspect
import stevedore
class TrueCheck(BaseCheck):
    """A policy check that always returns ``True`` (allow)."""

    def __str__(self):
        """Return a string representation of this check."""
        return '@'

    def __call__(self, target, cred, enforcer, current_rule=None):
        """Check the policy."""
        return True