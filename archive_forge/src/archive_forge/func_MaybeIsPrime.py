import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
def MaybeIsPrime(number):
    if FermatPrimalityTest(number) and MillerRabinPrimalityTest(number):
        return True
    else:
        return False