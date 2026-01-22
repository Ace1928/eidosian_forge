import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
class StandardTestSOCPs:

    @staticmethod
    def test_socp_0(solver, places: int=4, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = socp_0()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
        return sth

    @staticmethod
    def test_socp_1(solver, places: int=4, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = socp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_socp_2(solver, places: int=4, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = socp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_socp_3ax0(solver, places: int=3, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = socp_3(axis=0)
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_socp_3ax1(solver, places: int=3, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = socp_3(axis=1)
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_mi_socp_1(solver, places: int=4, **kwargs) -> SolverTestHelper:
        sth = mi_socp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth

    @staticmethod
    def test_mi_socp_2(solver, places: int=4, **kwargs) -> SolverTestHelper:
        sth = mi_socp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth