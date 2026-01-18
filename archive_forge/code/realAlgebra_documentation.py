from ..sage_helper import _within_sage, sage_method, SageNotAvailable

        Evaluation method for the factors of the equation factored over the
        number field. We take the factor and turn it into a polynomial in
        Q[x][y]. We then put in the given intervals for x and y.
        