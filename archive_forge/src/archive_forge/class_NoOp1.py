import numpy as np
import cirq
class NoOp1(EmptyOp):

    @property
    def gate(self):
        return No1()