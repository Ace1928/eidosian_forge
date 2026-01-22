import abc
import ast
import inspect
import stevedore
class AndCheck(BaseCheck):

    def __init__(self, rules):
        self.rules = rules

    def __str__(self):
        """Return a string representation of this check."""
        return '(%s)' % ' and '.join((str(r) for r in self.rules))

    def __call__(self, target, cred, enforcer, current_rule=None):
        """Check the policy.

        Requires that all rules accept in order to return True.
        """
        for rule in self.rules:
            if not _check(rule, target, cred, enforcer, current_rule):
                return False
        return True

    def add_check(self, rule):
        """Adds rule to be tested.

        Allows addition of another rule to the list of rules that will
        be tested.

        :returns: self
        :rtype: :class:`.AndCheck`
        """
        self.rules.append(rule)
        return self