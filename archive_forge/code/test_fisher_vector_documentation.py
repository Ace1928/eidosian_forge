import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402

    Test the improved Fisher vector computation given a GMM returned from the
    learn_gmm function. We simply assert that the dimensionality of the
    resulting Fisher vector is correct.

    The dimensionality of a Fisher vector is given by 2KD + K, where K is the
    number of Gaussians specified in the associated GMM, and D is the
    dimensionality of the descriptors using to estimate the GMM.
    