import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
class StandardTestPCPs:

    @staticmethod
    def test_pcp_1(solver, places: int=3, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = pcp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_pcp_2(solver, places: int=3, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = pcp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_pcp_3(solver, places: int=3, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = pcp_3()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)

    @staticmethod
    def test_mi_pcp_0(solver, places: int=3, **kwargs) -> SolverTestHelper:
        sth = mi_pcp_0()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        return sth