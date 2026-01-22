import numpy as np
import cirq
class NoOp2(EmptyOp):

    @property
    def gate(self):
        return No2()