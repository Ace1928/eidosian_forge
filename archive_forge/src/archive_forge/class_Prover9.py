import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
class Prover9(Prover9Parent, Prover):
    _prover9_bin = None
    _prooftrans_bin = None

    def __init__(self, timeout=60):
        self._timeout = timeout
        'The timeout value for prover9.  If a proof can not be found\n           in this amount of time, then prover9 will return false.\n           (Use 0 for no timeout.)'

    def _prove(self, goal=None, assumptions=None, verbose=False):
        """
        Use Prover9 to prove a theorem.
        :return: A pair whose first element is a boolean indicating if the
        proof was successful (i.e. returns value of 0) and whose second element
        is the output of the prover.
        """
        if not assumptions:
            assumptions = []
        stdout, returncode = self._call_prover9(self.prover9_input(goal, assumptions), verbose=verbose)
        return (returncode == 0, stdout)

    def prover9_input(self, goal, assumptions):
        """
        :see: Prover9Parent.prover9_input
        """
        s = 'clear(auto_denials).\n'
        return s + Prover9Parent.prover9_input(self, goal, assumptions)

    def _call_prover9(self, input_str, args=[], verbose=False):
        """
        Call the ``prover9`` binary with the given input.

        :param input_str: A string whose contents are used as stdin.
        :param args: A list of command-line arguments.
        :return: A tuple (stdout, returncode)
        :see: ``config_prover9``
        """
        if self._prover9_bin is None:
            self._prover9_bin = self._find_binary('prover9', verbose)
        updated_input_str = ''
        if self._timeout > 0:
            updated_input_str += 'assign(max_seconds, %d).\n\n' % self._timeout
        updated_input_str += input_str
        stdout, returncode = self._call(updated_input_str, self._prover9_bin, args, verbose)
        if returncode not in [0, 2]:
            errormsgprefix = '%%ERROR:'
            if errormsgprefix in stdout:
                msgstart = stdout.index(errormsgprefix)
                errormsg = stdout[msgstart:].strip()
            else:
                errormsg = None
            if returncode in [3, 4, 5, 6]:
                raise Prover9LimitExceededException(returncode, errormsg)
            else:
                raise Prover9FatalException(returncode, errormsg)
        return (stdout, returncode)

    def _call_prooftrans(self, input_str, args=[], verbose=False):
        """
        Call the ``prooftrans`` binary with the given input.

        :param input_str: A string whose contents are used as stdin.
        :param args: A list of command-line arguments.
        :return: A tuple (stdout, returncode)
        :see: ``config_prover9``
        """
        if self._prooftrans_bin is None:
            self._prooftrans_bin = self._find_binary('prooftrans', verbose)
        return self._call(input_str, self._prooftrans_bin, args, verbose)