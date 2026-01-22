from pyomo.core import Transformation, TransformationFactory

    Transform disjunctive model to equivalent MI(N)LP using the between steps
    transformation from Konqvist et al. 2021 [1].

    This transformation first calls the 'gdp.partition_disjuncts'
    transformation, resulting in an equivalent GDP with the constraints
    partitioned, and then takes the hull reformulation of that model to get
    an algebraic model.

    References
    ----------
        [1] J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate
            Relaxations between big-M and Convex Hull Reformulations," 2021.
    