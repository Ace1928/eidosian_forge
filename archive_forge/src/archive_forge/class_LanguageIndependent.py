import math
class LanguageIndependent:
    PRIORS = {(1, 0): 0.0099, (0, 1): 0.0099, (1, 1): 0.89, (2, 1): 0.089, (1, 2): 0.089, (2, 2): 0.011}
    AVERAGE_CHARACTERS = 1
    VARIANCE_CHARACTERS = 6.8