from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
def _solve_hinge_beams(self, *reactions):
    """Method to find integration constants and reactional variables in a
        composite beam connected via hinge.
        This method resolves the composite Beam into its sub-beams and then
        equations of shear force, bending moment, slope and deflection are
        evaluated for both of them separately. These equations are then solved
        for unknown reactions and integration constants using the boundary
        conditions applied on the Beam. Equal deflection of both sub-beams
        at the hinge joint gives us another equation to solve the system.

        Examples
        ========
        A combined beam, with constant fkexural rigidity E*I, is formed by joining
        a Beam of length 2*l to the right of another Beam of length l. The whole beam
        is fixed at both of its both end. A point load of magnitude P is also applied
        from the top at a distance of 2*l from starting point.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> l=symbols('l', positive=True)
        >>> b1=Beam(l, E, I)
        >>> b2=Beam(2*l, E, I)
        >>> b=b1.join(b2,"hinge")
        >>> M1, A1, M2, A2, P = symbols('M1 A1 M2 A2 P')
        >>> b.apply_load(A1,0,-1)
        >>> b.apply_load(M1,0,-2)
        >>> b.apply_load(P,2*l,-1)
        >>> b.apply_load(A2,3*l,-1)
        >>> b.apply_load(M2,3*l,-2)
        >>> b.bc_slope=[(0,0), (3*l, 0)]
        >>> b.bc_deflection=[(0,0), (3*l, 0)]
        >>> b.solve_for_reaction_loads(M1, A1, M2, A2)
        >>> b.reaction_loads
        {A1: -5*P/18, A2: -13*P/18, M1: 5*P*l/18, M2: -4*P*l/9}
        >>> b.slope()
        (5*P*l*SingularityFunction(x, 0, 1)/18 - 5*P*SingularityFunction(x, 0, 2)/36 + 5*P*SingularityFunction(x, l, 2)/36)*SingularityFunction(x, 0, 0)/(E*I)
        - (5*P*l*SingularityFunction(x, 0, 1)/18 - 5*P*SingularityFunction(x, 0, 2)/36 + 5*P*SingularityFunction(x, l, 2)/36)*SingularityFunction(x, l, 0)/(E*I)
        + (P*l**2/18 - 4*P*l*SingularityFunction(-l + x, 2*l, 1)/9 - 5*P*SingularityFunction(-l + x, 0, 2)/36 + P*SingularityFunction(-l + x, l, 2)/2
        - 13*P*SingularityFunction(-l + x, 2*l, 2)/36)*SingularityFunction(x, l, 0)/(E*I)
        >>> b.deflection()
        (5*P*l*SingularityFunction(x, 0, 2)/36 - 5*P*SingularityFunction(x, 0, 3)/108 + 5*P*SingularityFunction(x, l, 3)/108)*SingularityFunction(x, 0, 0)/(E*I)
        - (5*P*l*SingularityFunction(x, 0, 2)/36 - 5*P*SingularityFunction(x, 0, 3)/108 + 5*P*SingularityFunction(x, l, 3)/108)*SingularityFunction(x, l, 0)/(E*I)
        + (5*P*l**3/54 + P*l**2*(-l + x)/18 - 2*P*l*SingularityFunction(-l + x, 2*l, 2)/9 - 5*P*SingularityFunction(-l + x, 0, 3)/108 + P*SingularityFunction(-l + x, l, 3)/6
        - 13*P*SingularityFunction(-l + x, 2*l, 3)/108)*SingularityFunction(x, l, 0)/(E*I)
        """
    x = self.variable
    l = self._hinge_position
    E = self._elastic_modulus
    I = self._second_moment
    if isinstance(I, Piecewise):
        I1 = I.args[0][0]
        I2 = I.args[1][0]
    else:
        I1 = I2 = I
    load_1 = 0
    load_2 = 0
    for load in self.applied_loads:
        if load[1] < l:
            load_1 += load[0] * SingularityFunction(x, load[1], load[2])
            if load[2] == 0:
                load_1 -= load[0] * SingularityFunction(x, load[3], load[2])
            elif load[2] > 0:
                load_1 -= load[0] * SingularityFunction(x, load[3], load[2]) + load[0] * SingularityFunction(x, load[3], 0)
        elif load[1] == l:
            load_1 += load[0] * SingularityFunction(x, load[1], load[2])
            load_2 += load[0] * SingularityFunction(x, load[1] - l, load[2])
        elif load[1] > l:
            load_2 += load[0] * SingularityFunction(x, load[1] - l, load[2])
            if load[2] == 0:
                load_2 -= load[0] * SingularityFunction(x, load[3] - l, load[2])
            elif load[2] > 0:
                load_2 -= load[0] * SingularityFunction(x, load[3] - l, load[2]) + load[0] * SingularityFunction(x, load[3] - l, 0)
    h = Symbol('h')
    load_1 += h * SingularityFunction(x, l, -1)
    load_2 -= h * SingularityFunction(x, 0, -1)
    eq = []
    shear_1 = integrate(load_1, x)
    shear_curve_1 = limit(shear_1, x, l)
    eq.append(shear_curve_1)
    bending_1 = integrate(shear_1, x)
    moment_curve_1 = limit(bending_1, x, l)
    eq.append(moment_curve_1)
    shear_2 = integrate(load_2, x)
    shear_curve_2 = limit(shear_2, x, self.length - l)
    eq.append(shear_curve_2)
    bending_2 = integrate(shear_2, x)
    moment_curve_2 = limit(bending_2, x, self.length - l)
    eq.append(moment_curve_2)
    C1 = Symbol('C1')
    C2 = Symbol('C2')
    C3 = Symbol('C3')
    C4 = Symbol('C4')
    slope_1 = S.One / (E * I1) * (integrate(bending_1, x) + C1)
    def_1 = S.One / (E * I1) * (integrate(E * I * slope_1, x) + C1 * x + C2)
    slope_2 = S.One / (E * I2) * (integrate(integrate(integrate(load_2, x), x), x) + C3)
    def_2 = S.One / (E * I2) * (integrate(E * I * slope_2, x) + C4)
    for position, value in self.bc_slope:
        if position < l:
            eq.append(slope_1.subs(x, position) - value)
        else:
            eq.append(slope_2.subs(x, position - l) - value)
    for position, value in self.bc_deflection:
        if position < l:
            eq.append(def_1.subs(x, position) - value)
        else:
            eq.append(def_2.subs(x, position - l) - value)
    eq.append(def_1.subs(x, l) - def_2.subs(x, 0))
    constants = list(linsolve(eq, C1, C2, C3, C4, h, *reactions))
    reaction_values = list(constants[0])[5:]
    self._reaction_loads = dict(zip(reactions, reaction_values))
    self._load = self._load.subs(self._reaction_loads)
    slope_1 = slope_1.subs({C1: constants[0][0], h: constants[0][4]}).subs(self._reaction_loads)
    def_1 = def_1.subs({C1: constants[0][0], C2: constants[0][1], h: constants[0][4]}).subs(self._reaction_loads)
    slope_2 = slope_2.subs({x: x - l, C3: constants[0][2], h: constants[0][4]}).subs(self._reaction_loads)
    def_2 = def_2.subs({x: x - l, C3: constants[0][2], C4: constants[0][3], h: constants[0][4]}).subs(self._reaction_loads)
    self._hinge_beam_slope = slope_1 * SingularityFunction(x, 0, 0) - slope_1 * SingularityFunction(x, l, 0) + slope_2 * SingularityFunction(x, l, 0)
    self._hinge_beam_deflection = def_1 * SingularityFunction(x, 0, 0) - def_1 * SingularityFunction(x, l, 0) + def_2 * SingularityFunction(x, l, 0)