from sympy.core.basic import Basic
from sympy.stats.joint_rv import ProductPSpace
from sympy.stats.rv import ProductDomain, _symbol_converter, Distribution

        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        