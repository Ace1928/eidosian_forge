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
class Beam3D(Beam):
    """
    This class handles loads applied in any direction of a 3D space along
    with unequal values of Second moment along different axes.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention.
       This class assumes that any kind of distributed load/moment is
       applied through out the span of a beam.

    Examples
    ========
    There is a beam of l meters long. A constant distributed load of magnitude q
    is applied along y-axis from start till the end of beam. A constant distributed
    moment of magnitude m is also applied along z-axis from start till the end of beam.
    Beam is fixed at both of its end. So, deflection of the beam at the both ends
    is restricted.

    >>> from sympy.physics.continuum_mechanics.beam import Beam3D
    >>> from sympy import symbols, simplify, collect, factor
    >>> l, E, G, I, A = symbols('l, E, G, I, A')
    >>> b = Beam3D(l, E, G, I, A)
    >>> x, q, m = symbols('x, q, m')
    >>> b.apply_load(q, 0, 0, dir="y")
    >>> b.apply_moment_load(m, 0, -1, dir="z")
    >>> b.shear_force()
    [0, -q*x, 0]
    >>> b.bending_moment()
    [0, 0, -m*x + q*x**2/2]
    >>> b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.solve_slope_deflection()
    >>> factor(b.slope())
    [0, 0, x*(-l + x)*(-A*G*l**3*q + 2*A*G*l**2*q*x - 12*E*I*l*q
        - 72*E*I*m + 24*E*I*q*x)/(12*E*I*(A*G*l**2 + 12*E*I))]
    >>> dx, dy, dz = b.deflection()
    >>> dy = collect(simplify(dy), x)
    >>> dx == dz == 0
    True
    >>> dy == (x*(12*E*I*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q)
    ... + x*(A*G*l*(3*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q) + x*(-2*A*G*l**2*q + 4*A*G*l*m - 24*E*I*q))
    ... + A*G*(A*G*l**2 + 12*E*I)*(-2*l**2*q + 6*l*m - 4*m*x + q*x**2)
    ... - 12*E*I*q*(A*G*l**2 + 12*E*I)))/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
    True

    References
    ==========

    .. [1] https://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf

    """

    def __init__(self, length, elastic_modulus, shear_modulus, second_moment, area, variable=Symbol('x')):
        """Initializes the class.

        Parameters
        ==========
        length : Sympifyable
            A Symbol or value representing the Beam's length.
        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material.
        shear_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of rigidity.
            It is a measure of rigidity of the Beam material.
        second_moment : Sympifyable or list
            A list of two elements having SymPy expression representing the
            Beam's Second moment of area. First value represent Second moment
            across y-axis and second across z-axis.
            Single SymPy expression can be passed if both values are same
        area : Sympifyable
            A SymPy expression representing the Beam's cross-sectional area
            in a plane perpendicular to length of the Beam.
        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope and deflection
            curve. By default, it is set to ``Symbol('x')``.
        """
        super().__init__(length, elastic_modulus, second_moment, variable)
        self.shear_modulus = shear_modulus
        self.area = area
        self._load_vector = [0, 0, 0]
        self._moment_load_vector = [0, 0, 0]
        self._torsion_moment = {}
        self._load_Singularity = [0, 0, 0]
        self._slope = [0, 0, 0]
        self._deflection = [0, 0, 0]
        self._angular_deflection = 0

    @property
    def shear_modulus(self):
        """Young's Modulus of the Beam. """
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, e):
        self._shear_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        if isinstance(i, list):
            i = [sympify(x) for x in i]
            self._second_moment = i
        else:
            self._second_moment = sympify(i)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        return self._area

    @area.setter
    def area(self, a):
        self._area = sympify(a)

    @property
    def load_vector(self):
        """
        Returns a three element list representing the load vector.
        """
        return self._load_vector

    @property
    def moment_load_vector(self):
        """
        Returns a three element list representing moment loads on Beam.
        """
        return self._moment_load_vector

    @property
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has two keywords namely slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value). Further each value is a list corresponding to
        slope or deflection(s) values along three axes at that location.

        Examples
        ========
        There is a beam of length 4 meters. The slope at 0 should be 4 along
        the x-axis and 0 along others. At the other end of beam, deflection
        along all the three axes should be zero.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.bc_slope = [(0, (4, 0, 0))]
        >>> b.bc_deflection = [(4, [0, 0, 0])]
        >>> b.boundary_conditions
        {'deflection': [(4, [0, 0, 0])], 'slope': [(0, (4, 0, 0))]}

        Here the deflection of the beam should be ``0`` along all the three axes at ``4``.
        Similarly, the slope of the beam should be ``4`` along x-axis and ``0``
        along y and z axis at ``0``.
        """
        return self._boundary_conditions

    def polar_moment(self):
        """
        Returns the polar moment of area of the beam
        about the X axis with respect to the centroid.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A = symbols('l, E, G, I, A')
        >>> b = Beam3D(l, E, G, I, A)
        >>> b.polar_moment()
        2*I
        >>> I1 = [9, 15]
        >>> b = Beam3D(l, E, G, I1, A)
        >>> b.polar_moment()
        24
        """
        if not iterable(self.second_moment):
            return 2 * self.second_moment
        return sum(self.second_moment)

    def apply_load(self, value, start, order, dir='y'):
        """
        This method adds up the force load to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        dir : String
            Axis along which load is applied.
        order : Integer
            The order of the applied load.
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)
        if dir == 'x':
            if not order == -1:
                self._load_vector[0] += value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)
        elif dir == 'y':
            if not order == -1:
                self._load_vector[1] += value
            self._load_Singularity[1] += value * SingularityFunction(x, start, order)
        else:
            if not order == -1:
                self._load_vector[2] += value
            self._load_Singularity[2] += value * SingularityFunction(x, start, order)

    def apply_moment_load(self, value, start, order, dir='y'):
        """
        This method adds up the moment loads to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied moment.
        dir : String
            Axis along which moment is applied.
        order : Integer
            The order of the applied load.
            - For point moments, order=-2
            - For constant distributed moment, order=-1
            - For ramp moments, order=0
            - For parabolic ramp moments, order=1
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)
        if dir == 'x':
            if not order == -2:
                self._moment_load_vector[0] += value
            elif start in list(self._torsion_moment):
                self._torsion_moment[start] += value
            else:
                self._torsion_moment[start] = value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)
        elif dir == 'y':
            if not order == -2:
                self._moment_load_vector[1] += value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)
        else:
            if not order == -2:
                self._moment_load_vector[2] += value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)

    def apply_support(self, loc, type='fixed'):
        if type in ('pin', 'roller'):
            reaction_load = Symbol('R_' + str(loc))
            self._reaction_loads[reaction_load] = reaction_load
            self.bc_deflection.append((loc, [0, 0, 0]))
        else:
            reaction_load = Symbol('R_' + str(loc))
            reaction_moment = Symbol('M_' + str(loc))
            self._reaction_loads[reaction_load] = [reaction_load, reaction_moment]
            self.bc_deflection.append((loc, [0, 0, 0]))
            self.bc_slope.append((loc, [0, 0, 0]))

    def solve_for_reaction_loads(self, *reaction):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. It it supported by rollers at
        of its end. A constant distributed load of magnitude 8 N is applied
        from start till its end along y-axis. Another linear load having
        slope equal to 9 is applied along z-axis.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.apply_load(8, start=0, order=0, dir="y")
        >>> b.apply_load(9*x, start=0, order=0, dir="z")
        >>> b.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
        >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
        >>> b.apply_load(R1, start=0, order=-1, dir="y")
        >>> b.apply_load(R2, start=30, order=-1, dir="y")
        >>> b.apply_load(R3, start=0, order=-1, dir="z")
        >>> b.apply_load(R4, start=30, order=-1, dir="z")
        >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
        >>> b.reaction_loads
        {R1: -120, R2: -120, R3: -1350, R4: -2700}
        """
        x = self.variable
        l = self.length
        q = self._load_Singularity
        shear_curves = [integrate(load, x) for load in q]
        moment_curves = [integrate(shear, x) for shear in shear_curves]
        for i in range(3):
            react = [r for r in reaction if shear_curves[i].has(r) or moment_curves[i].has(r)]
            if len(react) == 0:
                continue
            shear_curve = limit(shear_curves[i], x, l)
            moment_curve = limit(moment_curves[i], x, l)
            sol = list(linsolve([shear_curve, moment_curve], react).args[0])
            sol_dict = dict(zip(react, sol))
            reaction_loads = self._reaction_loads
            for key in sol_dict:
                if key in reaction_loads and sol_dict[key] != reaction_loads[key]:
                    raise ValueError('Ambiguous solution for %s in different directions.' % key)
            self._reaction_loads.update(sol_dict)

    def shear_force(self):
        """
        Returns a list of three expressions which represents the shear force
        curve of the Beam object along all three axes.
        """
        x = self.variable
        q = self._load_vector
        return [integrate(-q[0], x), integrate(-q[1], x), integrate(-q[2], x)]

    def axial_force(self):
        """
        Returns expression of Axial shear force present inside the Beam object.
        """
        return self.shear_force()[0]

    def shear_stress(self):
        """
        Returns a list of three expressions which represents the shear stress
        curve of the Beam object along all three axes.
        """
        return [self.shear_force()[0] / self._area, self.shear_force()[1] / self._area, self.shear_force()[2] / self._area]

    def axial_stress(self):
        """
        Returns expression of Axial stress present inside the Beam object.
        """
        return self.axial_force() / self._area

    def bending_moment(self):
        """
        Returns a list of three expressions which represents the bending moment
        curve of the Beam object along all three axes.
        """
        x = self.variable
        m = self._moment_load_vector
        shear = self.shear_force()
        return [integrate(-m[0], x), integrate(-m[1] + shear[2], x), integrate(-m[2] - shear[1], x)]

    def torsional_moment(self):
        """
        Returns expression of Torsional moment present inside the Beam object.
        """
        return self.bending_moment()[0]

    def solve_for_torsion(self):
        """
        Solves for the angular deflection due to the torsional effects of
        moments being applied in the x-direction i.e. out of or into the beam.

        Here, a positive torque means the direction of the torque is positive
        i.e. out of the beam along the beam-axis. Likewise, a negative torque
        signifies a torque into the beam cross-section.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(20, E, G, I, A, x)
        >>> b.apply_moment_load(4, 4, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.solve_for_torsion()
        >>> b.angular_deflection().subs(x, 3)
        18/(G*I)
        """
        x = self.variable
        sum_moments = 0
        for point in list(self._torsion_moment):
            sum_moments += self._torsion_moment[point]
        list(self._torsion_moment).sort()
        pointsList = list(self._torsion_moment)
        torque_diagram = Piecewise((sum_moments, x <= pointsList[0]), (0, x >= pointsList[0]))
        for i in range(len(pointsList))[1:]:
            sum_moments -= self._torsion_moment[pointsList[i - 1]]
            torque_diagram += Piecewise((0, x <= pointsList[i - 1]), (sum_moments, x <= pointsList[i]), (0, x >= pointsList[i]))
        integrated_torque_diagram = integrate(torque_diagram)
        self._angular_deflection = integrated_torque_diagram / (self.shear_modulus * self.polar_moment())

    def solve_slope_deflection(self):
        x = self.variable
        l = self.length
        E = self.elastic_modulus
        G = self.shear_modulus
        I = self.second_moment
        if isinstance(I, list):
            I_y, I_z = (I[0], I[1])
        else:
            I_y = I_z = I
        A = self._area
        load = self._load_vector
        moment = self._moment_load_vector
        defl = Function('defl')
        theta = Function('theta')
        eq = Derivative(E * A * Derivative(defl(x), x), x) + load[0]
        def_x = dsolve(Eq(eq, 0), defl(x)).args[1]
        C1 = Symbol('C1')
        C2 = Symbol('C2')
        constants = list(linsolve([def_x.subs(x, 0), def_x.subs(x, l)], C1, C2).args[0])
        def_x = def_x.subs({C1: constants[0], C2: constants[1]})
        slope_x = def_x.diff(x)
        self._deflection[0] = def_x
        self._slope[0] = slope_x
        C_i = Symbol('C_i')
        eq1 = Derivative(E * I_z * Derivative(theta(x), x), x) + (integrate(-load[1], x) + C_i) + moment[2]
        slope_z = dsolve(Eq(eq1, 0)).args[1]
        constants = list(linsolve([slope_z.subs(x, 0), slope_z.subs(x, l)], C1, C2).args[0])
        slope_z = slope_z.subs({C1: constants[0], C2: constants[1]})
        eq2 = G * A * Derivative(defl(x), x) + load[1] * x - C_i - G * A * slope_z
        def_y = dsolve(Eq(eq2, 0), defl(x)).args[1]
        constants = list(linsolve([def_y.subs(x, 0), def_y.subs(x, l)], C1, C_i).args[0])
        self._deflection[1] = def_y.subs({C1: constants[0], C_i: constants[1]})
        self._slope[2] = slope_z.subs(C_i, constants[1])
        eq1 = Derivative(E * I_y * Derivative(theta(x), x), x) + (integrate(load[2], x) - C_i) + moment[1]
        slope_y = dsolve(Eq(eq1, 0)).args[1]
        constants = list(linsolve([slope_y.subs(x, 0), slope_y.subs(x, l)], C1, C2).args[0])
        slope_y = slope_y.subs({C1: constants[0], C2: constants[1]})
        eq2 = G * A * Derivative(defl(x), x) + load[2] * x - C_i + G * A * slope_y
        def_z = dsolve(Eq(eq2, 0)).args[1]
        constants = list(linsolve([def_z.subs(x, 0), def_z.subs(x, l)], C1, C_i).args[0])
        self._deflection[2] = def_z.subs({C1: constants[0], C_i: constants[1]})
        self._slope[1] = slope_y.subs(C_i, constants[1])

    def slope(self):
        """
        Returns a three element list representing slope of deflection curve
        along all the three axes.
        """
        return self._slope

    def deflection(self):
        """
        Returns a three element list representing deflection curve along all
        the three axes.
        """
        return self._deflection

    def angular_deflection(self):
        """
        Returns a function in x depicting how the angular deflection, due to moments
        in the x-axis on the beam, varies with x.
        """
        return self._angular_deflection

    def _plot_shear_force(self, dir, subs=None):
        shear_force = self.shear_force()
        if dir == 'x':
            dir_num = 0
            color = 'r'
        elif dir == 'y':
            dir_num = 1
            color = 'g'
        elif dir == 'z':
            dir_num = 2
            color = 'b'
        if subs is None:
            subs = {}
        for sym in shear_force[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_force[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Shear Force along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{V(%c)}$' % dir, line_color=color)

    def plot_shear_force(self, dir='all', subs=None):
        """

        Returns a plot for Shear force along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear force plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_force()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x for x over (0.0, 20.0)

        """
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_shear_force('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_shear_force('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_shear_force('z', subs)
            return Pz.show()
        else:
            Px = self._plot_shear_force('x', subs)
            Py = self._plot_shear_force('y', subs)
            Pz = self._plot_shear_force('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_bending_moment(self, dir, subs=None):
        bending_moment = self.bending_moment()
        if dir == 'x':
            dir_num = 0
            color = 'g'
        elif dir == 'y':
            dir_num = 1
            color = 'c'
        elif dir == 'z':
            dir_num = 2
            color = 'm'
        if subs is None:
            subs = {}
        for sym in bending_moment[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(bending_moment[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Bending Moment along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{M(%c)}$' % dir, line_color=color)

    def plot_bending_moment(self, dir='all', subs=None):
        """

        Returns a plot for bending moment along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which bending moment plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_bending_moment()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: 2*x**3 for x over (0.0, 20.0)

        """
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_bending_moment('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_bending_moment('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_bending_moment('z', subs)
            return Pz.show()
        else:
            Px = self._plot_bending_moment('x', subs)
            Py = self._plot_bending_moment('y', subs)
            Pz = self._plot_bending_moment('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_slope(self, dir, subs=None):
        slope = self.slope()
        if dir == 'x':
            dir_num = 0
            color = 'b'
        elif dir == 'y':
            dir_num = 1
            color = 'm'
        elif dir == 'z':
            dir_num = 2
            color = 'g'
        if subs is None:
            subs = {}
        for sym in slope[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(slope[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Slope along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{\\theta(%c)}$' % dir, line_color=color)

    def plot_slope(self, dir='all', subs=None):
        """

        Returns a plot for Slope along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which Slope plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_slope()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/8000 - 19*x**2/172 + 52*x/43 for x over (0.0, 20.0)

        """
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_slope('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_slope('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_slope('z', subs)
            return Pz.show()
        else:
            Px = self._plot_slope('x', subs)
            Py = self._plot_slope('y', subs)
            Pz = self._plot_slope('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_deflection(self, dir, subs=None):
        deflection = self.deflection()
        if dir == 'x':
            dir_num = 0
            color = 'm'
        elif dir == 'y':
            dir_num = 1
            color = 'r'
        elif dir == 'z':
            dir_num = 2
            color = 'c'
        if subs is None:
            subs = {}
        for sym in deflection[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(deflection[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Deflection along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{\\delta(%c)}$' % dir, line_color=color)

    def plot_deflection(self, dir='all', subs=None):
        """

        Returns a plot for Deflection along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which deflection plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_deflection()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/6400 - x**3/160 + 27*x**2/560 + 2*x/7 for x over (0.0, 20.0)


        """
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_deflection('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_deflection('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_deflection('z', subs)
            return Pz.show()
        else:
            Px = self._plot_deflection('x', subs)
            Py = self._plot_deflection('y', subs)
            Pz = self._plot_deflection('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def plot_loading_results(self, dir='x', subs=None):
        """

        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object along the direction specified.

        Parameters
        ==========

        dir : string (default : "x")
               Direction along which plots are required.
               If no direction is specified, plots along x-axis are displayed.
        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> subs = {E:40, G:21, I:100, A:25}
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_loading_results('y',subs)
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[3]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)

        """
        dir = dir.lower()
        if subs is None:
            subs = {}
        ax1 = self._plot_shear_force(dir, subs)
        ax2 = self._plot_bending_moment(dir, subs)
        ax3 = self._plot_slope(dir, subs)
        ax4 = self._plot_deflection(dir, subs)
        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _plot_shear_stress(self, dir, subs=None):
        shear_stress = self.shear_stress()
        if dir == 'x':
            dir_num = 0
            color = 'r'
        elif dir == 'y':
            dir_num = 1
            color = 'g'
        elif dir == 'z':
            dir_num = 2
            color = 'b'
        if subs is None:
            subs = {}
        for sym in shear_stress[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_stress[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Shear stress along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\tau(%c)$' % dir, line_color=color)

    def plot_shear_stress(self, dir='all', subs=None):
        """

        Returns a plot for Shear Stress along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear stress plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters and area of cross section 2 square
        meters. It it supported by rollers at of its end. A linear load having
        slope equal to 12 is applied along y-axis. A constant distributed load
        of magnitude 15 N is applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, 2, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_stress()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -3*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x/2 for x over (0.0, 20.0)

        """
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_shear_stress('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_shear_stress('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_shear_stress('z', subs)
            return Pz.show()
        else:
            Px = self._plot_shear_stress('x', subs)
            Py = self._plot_shear_stress('y', subs)
            Pz = self._plot_shear_stress('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _max_shear_force(self, dir):
        """
        Helper function for max_shear_force().
        """
        dir = dir.lower()
        if dir == 'x':
            dir_num = 0
        elif dir == 'y':
            dir_num = 1
        elif dir == 'z':
            dir_num = 2
        if not self.shear_force()[dir_num]:
            return (0, 0)
        load_curve = Piecewise((float('nan'), self.variable <= 0), (self._load_vector[dir_num], self.variable < self.length), (float('nan'), True))
        points = solve(load_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)
        points.append(self.length)
        shear_curve = self.shear_force()[dir_num]
        shear_values = [shear_curve.subs(self.variable, x) for x in points]
        shear_values = list(map(abs, shear_values))
        max_shear = max(shear_values)
        return (points[shear_values.index(max_shear)], max_shear)

    def max_shear_force(self):
        """
        Returns point of max shear force and its corresponding shear value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_shear_force()
            [(0, 0), (20, 2400), (20, 300)]
        """
        max_shear = []
        max_shear.append(self._max_shear_force('x'))
        max_shear.append(self._max_shear_force('y'))
        max_shear.append(self._max_shear_force('z'))
        return max_shear

    def _max_bending_moment(self, dir):
        """
        Helper function for max_bending_moment().
        """
        dir = dir.lower()
        if dir == 'x':
            dir_num = 0
        elif dir == 'y':
            dir_num = 1
        elif dir == 'z':
            dir_num = 2
        if not self.bending_moment()[dir_num]:
            return (0, 0)
        shear_curve = Piecewise((float('nan'), self.variable <= 0), (self.shear_force()[dir_num], self.variable < self.length), (float('nan'), True))
        points = solve(shear_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)
        points.append(self.length)
        bending_moment_curve = self.bending_moment()[dir_num]
        bending_moments = [bending_moment_curve.subs(self.variable, x) for x in points]
        bending_moments = list(map(abs, bending_moments))
        max_bending_moment = max(bending_moments)
        return (points[bending_moments.index(max_bending_moment)], max_bending_moment)

    def max_bending_moment(self):
        """
        Returns point of max bending moment and its corresponding bending moment value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_bending_moment()
            [(0, 0), (20, 3000), (20, 16000)]
        """
        max_bmoment = []
        max_bmoment.append(self._max_bending_moment('x'))
        max_bmoment.append(self._max_bending_moment('y'))
        max_bmoment.append(self._max_bending_moment('z'))
        return max_bmoment
    max_bmoment = max_bending_moment

    def _max_deflection(self, dir):
        """
        Helper function for max_Deflection()
        """
        dir = dir.lower()
        if dir == 'x':
            dir_num = 0
        elif dir == 'y':
            dir_num = 1
        elif dir == 'z':
            dir_num = 2
        if not self.deflection()[dir_num]:
            return (0, 0)
        slope_curve = Piecewise((float('nan'), self.variable <= 0), (self.slope()[dir_num], self.variable < self.length), (float('nan'), True))
        points = solve(slope_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)
        points.append(self._length)
        deflection_curve = self.deflection()[dir_num]
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))
        max_def = max(deflections)
        return (points[deflections.index(max_def)], max_def)

    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() and solve_slope_deflection() must be called
        before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It it supported by rollers
        at of its end. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.max_deflection()
            [(0, 0), (10, 495/14), (-10 + 10*sqrt(10793)/43, (10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560)]
        """
        max_def = []
        max_def.append(self._max_deflection('x'))
        max_def.append(self._max_deflection('y'))
        max_def.append(self._max_deflection('z'))
        return max_def