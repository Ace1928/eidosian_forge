import abc
import ast
import inspect
import stevedore
class NotCheck(BaseCheck):

    def __init__(self, rule):
        self.rule = rule

    def __str__(self):
        """Return a string representation of this check."""
        return 'not %s' % self.rule

    def __call__(self, target, cred, enforcer, current_rule=None):
        """Check the policy.

        Returns the logical inverse of the wrapped check.
        """
        return not _check(self.rule, target, cred, enforcer, current_rule)