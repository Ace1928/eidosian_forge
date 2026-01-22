import os
import math
import dill as pickle
def adder(augend):
    zero = [0]

    def inner(addend):
        return addend + augend + zero[0]
    return inner