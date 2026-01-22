import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
def FermatPrimalityTest(number):
    if number > 1:
        for time in range(3):
            randomNumber = random.randint(2, number) - 1
            if pow(randomNumber, number - 1, number) != 1:
                return False
        return True
    else:
        return False