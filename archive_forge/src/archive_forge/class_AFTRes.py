import numpy as np
class AFTRes:
    """
    Results for the AFT model from package emplik in R written by Mai Zhou
    """

    def __init__(self):
        self.test_params = np.array([3.77710799, -0.03281745])
        self.test_beta0 = (0.132511, 0.7158323)
        self.test_beta1 = (0.297951, 0.5851693)
        self.test_joint = (11.8068, 0.002730147)