import threading
import time
from abc import ABCMeta, abstractmethod
class ParallelProverBuilderCommand(BaseProverCommand, BaseModelBuilderCommand):
    """
    This command stores both a prover and a model builder and when either
    prove() or build_model() is called, then both theorem tools are run in
    parallel.  Whichever finishes first, the prover or the model builder, is the
    result that will be used.

    Because the theorem prover result is the opposite of the model builder
    result, we will treat self._result as meaning "proof found/no model found".
    """

    def __init__(self, prover, modelbuilder, goal=None, assumptions=None):
        BaseProverCommand.__init__(self, prover, goal, assumptions)
        BaseModelBuilderCommand.__init__(self, modelbuilder, goal, assumptions)

    def prove(self, verbose=False):
        return self._run(verbose)

    def build_model(self, verbose=False):
        return not self._run(verbose)

    def _run(self, verbose):
        tp_thread = TheoremToolThread(lambda: BaseProverCommand.prove(self, verbose), verbose, 'TP')
        mb_thread = TheoremToolThread(lambda: BaseModelBuilderCommand.build_model(self, verbose), verbose, 'MB')
        tp_thread.start()
        mb_thread.start()
        while tp_thread.is_alive() and mb_thread.is_alive():
            pass
        if tp_thread.result is not None:
            self._result = tp_thread.result
        elif mb_thread.result is not None:
            self._result = not mb_thread.result
        return self._result