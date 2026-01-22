import numpy as np
class LinearConstraints:
    """Class to hold linear constraints information

    Affine constraints are defined as ``R b = q` where `R` is the constraints
    matrix and `q` are the constraints values and `b` are the parameters.

    This is in analogy to patsy's LinearConstraints class but can be pickled.

    Parameters
    ----------
    constraint_matrix : ndarray
        R matrix, 2-dim with number of columns equal to the number of
        parameters. Each row defines one constraint.
    constraint_values : ndarray
        1-dim array of constant values
    variable_names : list of strings
        parameter names, used only for display
    kwds : keyword arguments
        keywords are attached to the instance.

    """

    def __init__(self, constraint_matrix, constraint_values, variable_names, **kwds):
        self.constraint_matrix = constraint_matrix
        self.constraint_values = constraint_values
        self.variable_names = variable_names
        self.coefs = constraint_matrix
        self.constants = constraint_values
        self.__dict__.update(kwds)
        self.tuple = (self.constraint_matrix, self.constraint_values)

    def __iter__(self):
        yield from self.tuple

    def __getitem__(self, idx):
        return self.tuple[idx]

    def __str__(self):

        def prod_string(v, name):
            v = np.abs(v)
            if v != 1:
                ss = str(v) + ' * ' + name
            else:
                ss = name
            return ss
        constraints_strings = []
        for r, q in zip(*self):
            ss = []
            for v, name in zip(r, self.variable_names):
                if v != 0 and ss == []:
                    ss += prod_string(v, name)
                elif v > 0:
                    ss += ' + ' + prod_string(v, name)
                elif v < 0:
                    ss += ' - ' + prod_string(np.abs(v), name)
            ss += ' = ' + str(q.item())
            constraints_strings.append(''.join(ss))
        return '\n'.join(constraints_strings)

    @classmethod
    def from_patsy(cls, lc):
        """class method to create instance from patsy instance

        Parameters
        ----------
        lc : instance
            instance of patsy LinearConstraint, or other instances that have
            attributes ``lc.coefs, lc.constants, lc.variable_names``

        Returns
        -------
        instance of this class

        """
        return cls(lc.coefs, lc.constants, lc.variable_names)