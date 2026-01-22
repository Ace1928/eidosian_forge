import threading
import time
from abc import ABCMeta, abstractmethod
class BaseTheoremToolCommand(TheoremToolCommand):
    """
    This class holds a goal and a list of assumptions to be used in proving
    or model building.
    """

    def __init__(self, goal=None, assumptions=None):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        """
        self._goal = goal
        if not assumptions:
            self._assumptions = []
        else:
            self._assumptions = list(assumptions)
        self._result = None
        'A holder for the result, to prevent unnecessary re-proving'

    def add_assumptions(self, new_assumptions):
        """
        Add new assumptions to the assumption list.

        :param new_assumptions: new assumptions
        :type new_assumptions: list(sem.Expression)
        """
        self._assumptions.extend(new_assumptions)
        self._result = None

    def retract_assumptions(self, retracted, debug=False):
        """
        Retract assumptions from the assumption list.

        :param debug: If True, give warning when ``retracted`` is not present on
            assumptions list.
        :type debug: bool
        :param retracted: assumptions to be retracted
        :type retracted: list(sem.Expression)
        """
        retracted = set(retracted)
        result_list = list(filter(lambda a: a not in retracted, self._assumptions))
        if debug and result_list == self._assumptions:
            print(Warning('Assumptions list has not been changed:'))
            self.print_assumptions()
        self._assumptions = result_list
        self._result = None

    def assumptions(self):
        """
        List the current assumptions.

        :return: list of ``Expression``
        """
        return self._assumptions

    def goal(self):
        """
        Return the goal

        :return: ``Expression``
        """
        return self._goal

    def print_assumptions(self):
        """
        Print the list of the current assumptions.
        """
        for a in self.assumptions():
            print(a)