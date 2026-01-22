import abc
import ast
import inspect
import stevedore
class OrCheck(BaseCheck):

    def __init__(self, rules):
        self.rules = rules

    def __str__(self):
        """Return a string representation of this check."""
        return '(%s)' % ' or '.join((str(r) for r in self.rules))

    def __call__(self, target, cred, enforcer, current_rule=None):
        """Check the policy.

        Requires that at least one rule accept in order to return True.
        """
        for rule in self.rules:
            if _check(rule, target, cred, enforcer, current_rule):
                return True
        return False

    def add_check(self, rule):
        """Adds rule to be tested.

        Allows addition of another rule to the list of rules that will
        be tested.  Returns the OrCheck object for convenience.
        """
        self.rules.append(rule)
        return self

    def pop_check(self):
        """Pops the last check from the list and returns them

        :returns: self, the popped check
        :rtype: :class:`.OrCheck`, class:`.Check`
        """
        check = self.rules.pop()
        return (self, check)