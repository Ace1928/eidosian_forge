import abc
Returns the Jacobian of the inequality constraints evaluated
        at the values given for the primal variables in set_primals

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        matrix_like
        