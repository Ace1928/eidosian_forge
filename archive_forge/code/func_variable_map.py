from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def variable_map(self, otherframe):
    """
        Returns a dictionary which expresses the coordinate variables
        of this frame in terms of the variables of otherframe.

        If Vector.simp is True, returns a simplified version of the mapped
        values. Else, returns them without simplification.

        Simplification of the expressions may take time.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The other frame to map the variables to

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> A = ReferenceFrame('A')
        >>> q = dynamicsymbols('q')
        >>> B = A.orientnew('B', 'Axis', [q, A.z])
        >>> A.variable_map(B)
        {A_x: B_x*cos(q(t)) - B_y*sin(q(t)), A_y: B_x*sin(q(t)) + B_y*cos(q(t)), A_z: B_z}

        """
    _check_frame(otherframe)
    if (otherframe, Vector.simp) in self._var_dict:
        return self._var_dict[otherframe, Vector.simp]
    else:
        vars_matrix = self.dcm(otherframe) * Matrix(otherframe.varlist)
        mapping = {}
        for i, x in enumerate(self):
            if Vector.simp:
                mapping[self.varlist[i]] = trigsimp(vars_matrix[i], method='fu')
            else:
                mapping[self.varlist[i]] = vars_matrix[i]
        self._var_dict[otherframe, Vector.simp] = mapping
        return mapping