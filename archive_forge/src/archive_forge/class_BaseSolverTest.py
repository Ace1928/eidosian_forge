import os
import tempfile
from pulp.constants import PulpError
from pulp.apis import *
from pulp import LpVariable, LpProblem, lpSum, LpConstraintVar, LpFractionConstraint
from pulp import constants as const
from pulp.tests.bin_packing_problem import create_bin_packing_problem
from pulp.utilities import makeDict
import functools
import unittest
class BaseSolverTest:

    class PuLPTest(unittest.TestCase):
        solveInst = None

        def setUp(self):
            self.solver = self.solveInst(msg=False)
            if not self.solver.available():
                self.skipTest(f'solver {self.solveInst} not available')

        def tearDown(self):
            for ext in ['mst', 'log', 'lp', 'mps', 'sol']:
                filename = f'{self._testMethodName}.{ext}'
                try:
                    os.remove(filename)
                except:
                    pass
            pass

        def test_pulp_001(self):
            """
            Test that a variable is deleted when it is suptracted to 0
            """
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            c1 = x + y <= 5
            c2 = c1 + z - z
            print('\t Testing zero subtraction')
            assert str(c2)

        def test_pulp_009(self):
            prob = LpProblem('test09', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (lpSum([v for v in [x] if False]) >= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing inconsistent lp solution')
            if self.solver.__class__ in [PULP_CBC_CMD, COIN_CMD]:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible], {x: 4, y: -1, z: 6, w: 0}, use_mps=False)
            elif self.solver.__class__ in [CHOCO_CMD, MIPCL_CMD]:
                pass
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusNotSolved, const.LpStatusUndefined])

        def test_pulp_010(self):
            prob = LpProblem('test010', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing continuous LP solution')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_pulp_011(self):
            prob = LpProblem('test011', const.LpMaximize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing maximize continuous LP solution')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: 1, z: 8, w: 0})

        def test_pulp_012(self):
            prob = LpProblem('test012', const.LpMaximize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z + w, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing unbounded continuous LP solution')
            if self.solver.__class__ in [GUROBI, CPLEX_CMD, YAPOSIB, MOSEK, COPT]:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUnbounded])
            elif self.solver.__class__ in [COINMP_DLL, MIPCL_CMD]:
                print('\t\t Error in CoinMP and MIPCL_CMD: reports Optimal')
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])
            elif self.solver.__class__ is GLPK_CMD:
                pulpTestCheck(prob, self.solver, [const.LpStatusUndefined])
            elif self.solver.__class__ in [GUROBI_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY]:
                pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
            elif self.solver.__class__ in [CHOCO_CMD]:
                pass
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusUnbounded])

        def test_pulp_013(self):
            prob = LpProblem('test013', const.LpMinimize)
            x = LpVariable('x' * 120, 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing Long Names')
            if self.solver.__class__ in [CPLEX_CMD, GLPK_CMD, GUROBI_CMD, MIPCL_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY, HiGHS, HiGHS_CMD, XPRESS, XPRESS_CMD]:
                try:
                    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})
                except PulpError:
                    pass
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_pulp_014(self):
            prob = LpProblem('test014', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('x', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing repeated Names')
            if self.solver.__class__ in [COIN_CMD, COINMP_DLL, PULP_CBC_CMD, CPLEX_CMD, CPLEX_PY, GLPK_CMD, GUROBI_CMD, CHOCO_CMD, MIPCL_CMD, MOSEK, SCIP_CMD, FSCIP_CMD, SCIP_PY, HiGHS, HiGHS_CMD, XPRESS, XPRESS_CMD, XPRESS_PY]:
                try:
                    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})
                except PulpError:
                    pass
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_pulp_015(self):
            prob = LpProblem('test015', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            prob += (lpSum([0, 0]) <= 0, 'c5')
            print('\t Testing zero constraint')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_pulp_016(self):
            prob = LpProblem('test016', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            prob += (lpSum([0, 0]) <= 0, 'c5')
            print('\t Testing zero objective')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])

        def test_pulp_017(self):
            prob = LpProblem('test017', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob.setObjective(x)
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            prob += (lpSum([0, 0]) <= 0, 'c5')
            print('\t Testing LpVariable (not LpAffineExpression) objective')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])

        def test_pulp_018(self):
            prob = LpProblem('test018', const.LpMinimize)
            x = LpVariable('x' * 90, 0, 4)
            y = LpVariable('y' * 90, -1, 1)
            z = LpVariable('z' * 90, 0)
            w = LpVariable('w' * 90, 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            if self.solver.__class__ in [PULP_CBC_CMD, COIN_CMD]:
                print('\t Testing Long lines in LP')
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0}, use_mps=False)

        def test_pulp_019(self):
            prob = LpProblem('test019', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += ((2 * x + 2 * y).__div__(2.0) <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing LpAffineExpression divide')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_pulp_020(self):
            prob = LpProblem('test020', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            print('\t Testing MIP solution')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3, y: -0.5, z: 7})

        def test_pulp_021(self):
            prob = LpProblem('test021', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (1.1 * x + 4.1 * y + 9.1 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            print('\t Testing MIP solution with floats in objective')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3, y: -0.5, z: 7}, objective=64.95)

        def test_pulp_022(self):
            prob = LpProblem('test022', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            x.setInitialValue(3)
            y.setInitialValue(-0.5)
            z.setInitialValue(7)
            if self.solver.name in ['GUROBI', 'GUROBI_CMD', 'CPLEX_CMD', 'CPLEX_PY', 'COPT', 'HiGHS_CMD']:
                self.solver.optionsDict['warmStart'] = True
            print('\t Testing Initial value in MIP solution')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3, y: -0.5, z: 7})

        def test_pulp_023(self):
            prob = LpProblem('test023', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            solution = {x: 4, y: -0.5, z: 7}
            for v in [x, y, z]:
                v.setInitialValue(solution[v])
                v.fixValue()
            self.solver.optionsDict['warmStart'] = True
            print('\t Testing fixing value in MIP solution')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], solution)

        def test_pulp_030(self):
            prob = LpProblem('test030', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            self.solver.mip = 0
            print('\t Testing MIP relaxation')
            if self.solver.__class__ in [GUROBI_CMD, CHOCO_CMD, MIPCL_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY]:
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3.0, y: -0.5, z: 7})
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3.5, y: -1, z: 6.5})

        def test_pulp_040(self):
            prob = LpProblem('test040', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            print('\t Testing feasibility problem (no objective)')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])

        def test_pulp_050(self):
            prob = LpProblem('test050', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, 10)
            prob += (x + y <= 5.2, 'c1')
            prob += (x + z >= 10.3, 'c2')
            prob += (-y + z == 17.5, 'c3')
            print('\t Testing an infeasible problem')
            if self.solver.__class__ is GLPK_CMD:
                pulpTestCheck(prob, self.solver, [const.LpStatusUndefined])
            elif self.solver.__class__ in [GUROBI_CMD, FSCIP_CMD]:
                pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible])

        def test_pulp_060(self):
            prob = LpProblem('test060', const.LpMinimize)
            x = LpVariable('x', 0, 4, const.LpInteger)
            y = LpVariable('y', -1, 1, const.LpInteger)
            z = LpVariable('z', 0, 10, const.LpInteger)
            prob += (x + y <= 5.2, 'c1')
            prob += (x + z >= 10.3, 'c2')
            prob += (-y + z == 7.4, 'c3')
            print('\t Testing an integer infeasible problem')
            if self.solver.__class__ in [GLPK_CMD, COIN_CMD, PULP_CBC_CMD, MOSEK]:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUndefined])
            elif self.solver.__class__ in [COINMP_DLL]:
                print('\t\t Error in CoinMP to be fixed, reports Optimal')
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])
            elif self.solver.__class__ in [GUROBI_CMD, FSCIP_CMD]:
                pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible])

        def test_pulp_061(self):
            prob = LpProblem('sample', const.LpMaximize)
            dummy = LpVariable('dummy')
            c1 = LpVariable('c1', 0, 1, const.LpBinary)
            c2 = LpVariable('c2', 0, 1, const.LpBinary)
            prob += dummy
            prob += c1 + c2 == 2
            prob += c1 <= 0
            print('\t Testing another integer infeasible problem')
            if self.solver.__class__ in [GUROBI_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY]:
                pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
            elif self.solver.__class__ in [GLPK_CMD]:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUndefined])
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible])

        def test_pulp_070(self):
            prob = LpProblem('test070', const.LpMinimize)
            obj = LpConstraintVar('obj')
            a = LpConstraintVar('C1', const.LpConstraintLE, 5)
            b = LpConstraintVar('C2', const.LpConstraintGE, 10)
            c = LpConstraintVar('C3', const.LpConstraintEQ, 7)
            prob.setObjective(obj)
            prob += a
            prob += b
            prob += c
            x = LpVariable('x', 0, 4, const.LpContinuous, obj + a + b)
            y = LpVariable('y', -1, 1, const.LpContinuous, 4 * obj + a - c)
            z = LpVariable('z', 0, None, const.LpContinuous, 9 * obj + b + c)
            print('\t Testing column based modelling')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6})

        def test_pulp_075(self):
            prob = LpProblem('test075', const.LpMinimize)
            obj = LpConstraintVar('obj')
            a = LpConstraintVar('C1', const.LpConstraintLE, 5)
            b = LpConstraintVar('C2', const.LpConstraintGE, 10)
            c = LpConstraintVar('C3', const.LpConstraintEQ, 7)
            prob.setObjective(obj)
            prob += a
            prob += b
            prob += c
            x = LpVariable('x', 0, 4, const.LpContinuous, obj + b)
            y = LpVariable('y', -1, 1, const.LpContinuous, 4 * obj - c)
            z = LpVariable('z', 0, None, const.LpContinuous, 9 * obj + b + c)
            if self.solver.__class__ in [CPLEX_CMD, COINMP_DLL, YAPOSIB, PYGLPK]:
                print('\t Testing column based modelling with empty constraints')
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6})

        def test_pulp_080(self):
            """
            Test the reporting of dual variables slacks and reduced costs
            """
            prob = LpProblem('test080', const.LpMinimize)
            x = LpVariable('x', 0, 5)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            c1 = x + y <= 5
            c2 = x + z >= 10
            c3 = -y + z == 7
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (c1, 'c1')
            prob += (c2, 'c2')
            prob += (c3, 'c3')
            if self.solver.__class__ in [CPLEX_CMD, COINMP_DLL, PULP_CBC_CMD, YAPOSIB, PYGLPK]:
                print('\t Testing dual variables and slacks reporting')
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], sol={x: 4, y: -1, z: 6}, reducedcosts={x: 0, y: 12, z: 0}, duals={'c1': 0, 'c2': 1, 'c3': 8}, slacks={'c1': 2, 'c2': 0, 'c3': 0})

        def test_pulp_090(self):
            prob = LpProblem('test090', const.LpMinimize)
            obj = LpConstraintVar('obj')
            a = LpConstraintVar('C1', const.LpConstraintLE, 5)
            b = LpConstraintVar('C2', const.LpConstraintGE, 10)
            c = LpConstraintVar('C3', const.LpConstraintEQ, 7)
            prob.setObjective(obj)
            prob += a
            prob += b
            prob += c
            prob.setSolver(self.solver)
            x = LpVariable('x', 0, 4, const.LpContinuous, obj + a + b)
            y = LpVariable('y', -1, 1, const.LpContinuous, 4 * obj + a - c)
            prob.resolve()
            z = LpVariable('z', 0, None, const.LpContinuous, 9 * obj + b + c)
            if self.solver.__class__ in [COINMP_DLL]:
                print('\t Testing resolve of problem')
                prob.resolve()

        def test_pulp_100(self):
            """
            Test the ability to sequentially solve a problem
            """
            prob = LpProblem('test100', const.LpMinimize)
            x = LpVariable('x', 0, 1)
            y = LpVariable('y', 0, 1)
            z = LpVariable('z', 0, 1)
            obj1 = x + 0 * y + 0 * z
            obj2 = 0 * x - 1 * y + 0 * z
            prob += (x <= 1, 'c1')
            if self.solver.__class__ in [COINMP_DLL, GUROBI]:
                print('\t Testing Sequential Solves')
                status = prob.sequentialSolve([obj1, obj2], solver=self.solver)
                pulpTestCheck(prob, self.solver, [[const.LpStatusOptimal, const.LpStatusOptimal]], sol={x: 0, y: 1}, status=status)

        def test_pulp_110(self):
            """
            Test the ability to use fractional constraints
            """
            prob = LpProblem('test110', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            prob += LpFractionConstraint(x, z, const.LpConstraintEQ, 0.5, name='c5')
            print('\t Testing fractional constraints')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 10 / 3.0, y: -1 / 3.0, z: 20 / 3.0, w: 0})

        def test_pulp_120(self):
            """
            Test the ability to use Elastic constraints
            """
            prob = LpProblem('test120', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w')
            prob += (x + 4 * y + 9 * z + w, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob.extend((w >= -1).makeElasticSubProblem())
            print('\t Testing elastic constraints (no change)')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: -1})

        def test_pulp_121(self):
            """
            Test the ability to use Elastic constraints
            """
            prob = LpProblem('test121', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w')
            prob += (x + 4 * y + 9 * z + w, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob.extend((w >= -1).makeElasticSubProblem(proportionFreeBound=0.1))
            print('\t Testing elastic constraints (freebound)')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: -1.1})

        def test_pulp_122(self):
            """
            Test the ability to use Elastic constraints (penalty unchanged)
            """
            prob = LpProblem('test122', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w')
            prob += (x + 4 * y + 9 * z + w, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob.extend((w >= -1).makeElasticSubProblem(penalty=1.1))
            print('\t Testing elastic constraints (penalty unchanged)')
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: -1.0})

        def test_pulp_123(self):
            """
            Test the ability to use Elastic constraints (penalty unbounded)
            """
            prob = LpProblem('test123', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w')
            prob += (x + 4 * y + 9 * z + w, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob.extend((w >= -1).makeElasticSubProblem(penalty=0.9))
            print('\t Testing elastic constraints (penalty unbounded)')
            if self.solver.__class__ in [COINMP_DLL, GUROBI, CPLEX_CMD, YAPOSIB, MOSEK, COPT]:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUnbounded])
            elif self.solver.__class__ is GLPK_CMD:
                pulpTestCheck(prob, self.solver, [const.LpStatusUndefined])
            elif self.solver.__class__ in [GUROBI_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY]:
                pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
            elif self.solver.__class__ in [CHOCO_CMD]:
                pass
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusUnbounded])

        def test_msg_arg(self):
            """
            Test setting the msg arg to True does not interfere with solve
            """
            prob = LpProblem('test_msg_arg', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            data = prob.toDict()
            var1, prob1 = LpProblem.fromDict(data)
            x, y, z, w = (var1[name] for name in ['x', 'y', 'z', 'w'])
            if self.solver.name in ['HiGHS']:
                return
            self.solver.msg = True
            pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_pulpTestAll(self):
            """
            Test the availability of the function pulpTestAll
            """
            print('\t Testing the availability of the function pulpTestAll')
            from pulp import pulpTestAll

        def test_export_dict_LP(self):
            prob = LpProblem('test_export_dict_LP', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            data = prob.toDict()
            var1, prob1 = LpProblem.fromDict(data)
            x, y, z, w = (var1[name] for name in ['x', 'y', 'z', 'w'])
            print('\t Testing continuous LP solution - export dict')
            pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_export_dict_LP_no_obj(self):
            prob = LpProblem('test_export_dict_LP_no_obj', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0, 0)
            prob += (x + y >= 5, 'c1')
            prob += (x + z == 10, 'c2')
            prob += (-y + z <= 7, 'c3')
            prob += (w >= 0, 'c4')
            data = prob.toDict()
            var1, prob1 = LpProblem.fromDict(data)
            x, y, z, w = (var1[name] for name in ['x', 'y', 'z', 'w'])
            print('\t Testing export dict for LP')
            pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 4, y: 1, z: 6, w: 0})

        def test_export_json_LP(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            filename = name + '.json'
            prob.toJson(filename, indent=4)
            var1, prob1 = LpProblem.fromJson(filename)
            try:
                os.remove(filename)
            except:
                pass
            x, y, z, w = (var1[name] for name in ['x', 'y', 'z', 'w'])
            print('\t Testing continuous LP solution - export JSON')
            pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_export_dict_MIP(self):
            import copy
            prob = LpProblem('test_export_dict_MIP', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            data = prob.toDict()
            data_backup = copy.deepcopy(data)
            var1, prob1 = LpProblem.fromDict(data)
            x, y, z = (var1[name] for name in ['x', 'y', 'z'])
            print('\t Testing export dict MIP')
            pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 3, y: -0.5, z: 7})
            self.assertDictEqual(data, data_backup)

        def test_export_dict_max(self):
            prob = LpProblem('test_export_dict_max', const.LpMaximize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            data = prob.toDict()
            var1, prob1 = LpProblem.fromDict(data)
            x, y, z, w = (var1[name] for name in ['x', 'y', 'z', 'w'])
            print('\t Testing maximize continuous LP solution')
            pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 4, y: 1, z: 8, w: 0})

        def test_export_solver_dict_LP(self):
            prob = LpProblem('test_export_dict_LP', const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            data = self.solver.toDict()
            solver1 = getSolverFromDict(data)
            print('\t Testing continuous LP solution - export solver dict')
            pulpTestCheck(prob, solver1, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_export_solver_json(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            self.solver.mip = True
            logFilename = name + '.log'
            if self.solver.name == 'CPLEX_CMD':
                self.solver.optionsDict = dict(gapRel=0.1, gapAbs=1, maxMemory=1000, maxNodes=1, threads=1, logPath=logFilename, warmStart=True)
            elif self.solver.name in ['GUROBI_CMD', 'COIN_CMD', 'PULP_CBC_CMD']:
                self.solver.optionsDict = dict(gapRel=0.1, gapAbs=1, threads=1, logPath=logFilename, warmStart=True)
            filename = name + '.json'
            self.solver.toJson(filename, indent=4)
            solver1 = getSolverFromJson(filename)
            try:
                os.remove(filename)
            except:
                pass
            print('\t Testing continuous LP solution - export solver JSON')
            pulpTestCheck(prob, solver1, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_timeLimit(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            self.solver.timeLimit = 20
            print('\t Testing timeLimit argument')
            if self.solver.name != 'CHOCO_CMD':
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_assignInvalidStatus(self):
            print('\t Testing invalid status')
            t = LpProblem('test')
            Invalid = -100
            self.assertRaises(const.PulpError, lambda: t.assignStatus(Invalid))
            self.assertRaises(const.PulpError, lambda: t.assignStatus(0, Invalid))

        def test_logPath(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            logFilename = name + '.log'
            self.solver.optionsDict['logPath'] = logFilename
            if self.solver.name in ['CPLEX_PY', 'CPLEX_CMD', 'GUROBI', 'GUROBI_CMD', 'PULP_CBC_CMD', 'COIN_CMD']:
                print('\t Testing logPath argument')
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})
                if not os.path.exists(logFilename):
                    raise PulpError(f'Test failed for solver: {self.solver}')
                if not os.path.getsize(logFilename):
                    raise PulpError(f'Test failed for solver: {self.solver}')

        def test_makeDict_behavior(self):
            """
            Test if makeDict is returning the expected value.
            """
            headers = [['A', 'B'], ['C', 'D']]
            values = [[1, 2], [3, 4]]
            target = {'A': {'C': 1, 'D': 2}, 'B': {'C': 3, 'D': 4}}
            dict_with_default = makeDict(headers, values, default=0)
            dict_without_default = makeDict(headers, values)
            print('\t Testing makeDict general behavior')
            self.assertEqual(dict_with_default, target)
            self.assertEqual(dict_without_default, target)

        def test_makeDict_default_value(self):
            """
            Test if makeDict is returning a default value when specified.
            """
            headers = [['A', 'B'], ['C', 'D']]
            values = [[1, 2], [3, 4]]
            dict_with_default = makeDict(headers, values, default=0)
            dict_without_default = makeDict(headers, values)
            print('\t Testing makeDict default value behavior')
            self.assertEqual(dict_with_default['X']['Y'], 0)
            _func = lambda: dict_without_default['X']['Y']
            self.assertRaises(KeyError, _func)

        def test_importMPS_maximize(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMaximize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            filename = name + '.mps'
            prob.writeMPS(filename)
            _vars, prob2 = LpProblem.fromMPS(filename, sense=prob.sense)
            _dict1 = getSortedDict(prob)
            _dict2 = getSortedDict(prob2)
            print('\t Testing reading MPS files - maximize')
            self.assertDictEqual(_dict1, _dict2)

        def test_importMPS_noname(self):
            name = self._testMethodName
            prob = LpProblem('', const.LpMaximize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            filename = name + '.mps'
            prob.writeMPS(filename)
            _vars, prob2 = LpProblem.fromMPS(filename, sense=prob.sense)
            _dict1 = getSortedDict(prob)
            _dict2 = getSortedDict(prob2)
            print('\t Testing reading MPS files - noname')
            self.assertDictEqual(_dict1, _dict2)

        def test_importMPS_integer(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0, None, const.LpInteger)
            prob += (1.1 * x + 4.1 * y + 9.1 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7.5, 'c3')
            filename = name + '.mps'
            prob.writeMPS(filename)
            _vars, prob2 = LpProblem.fromMPS(filename, sense=prob.sense)
            _dict1 = getSortedDict(prob)
            _dict2 = getSortedDict(prob2)
            print('\t Testing reading MPS files - integer variable')
            self.assertDictEqual(_dict1, _dict2)

        def test_importMPS_binary(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMaximize)
            dummy = LpVariable('dummy')
            c1 = LpVariable('c1', 0, 1, const.LpBinary)
            c2 = LpVariable('c2', 0, 1, const.LpBinary)
            prob += dummy
            prob += c1 + c2 == 2
            prob += c1 <= 0
            filename = name + '.mps'
            prob.writeMPS(filename)
            _vars, prob2 = LpProblem.fromMPS(filename, sense=prob.sense, dropConsNames=True)
            _dict1 = getSortedDict(prob, keyCons='constant')
            _dict2 = getSortedDict(prob2, keyCons='constant')
            print('\t Testing reading MPS files - binary variable, no constraint names')
            self.assertDictEqual(_dict1, _dict2)

        def test_importMPS_RHS_fields56(self):
            """Import MPS file with RHS definitions in fields 5 & 6."""
            with tempfile.NamedTemporaryFile(delete=False) as h:
                h.write(str.encode(EXAMPLE_MPS_RHS56))
            _, problem = LpProblem.fromMPS(h.name)
            os.unlink(h.name)
            self.assertEqual(problem.constraints['LIM2'].constant, -10)

        def test_unset_objective_value__is_valid(self):
            """Given a valid problem that does not converge,
            assert that it is still categorised as valid.
            """
            name = self._testMethodName
            prob = LpProblem(name, const.LpMaximize)
            x = LpVariable('x')
            prob += 0 * x
            prob += x >= 1
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])
            self.assertTrue(prob.valid())

        def test_unbounded_problem__is_not_valid(self):
            """Given an unbounded problem, where x will tend to infinity
            to maximise the objective, assert that it is categorised
            as invalid."""
            name = self._testMethodName
            prob = LpProblem(name, const.LpMaximize)
            x = LpVariable('x')
            prob += 1000 * x
            prob += x >= 1
            self.assertFalse(prob.valid())

        def test_infeasible_problem__is_not_valid(self):
            """Given a problem where x cannot converge to any value
            given conflicting constraints, assert that it is invalid."""
            name = self._testMethodName
            prob = LpProblem(name, const.LpMaximize)
            x = LpVariable('x')
            prob += 1 * x
            prob += x >= 2
            prob += x <= 1
            if self.solver.name in ['GUROBI_CMD', 'FSCIP_CMD']:
                pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved, const.LpStatusInfeasible, const.LpStatusUndefined])
            else:
                pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUndefined])
            self.assertFalse(prob.valid())

        def test_false_constraint(self):
            prob = LpProblem(self._testMethodName, const.LpMinimize)

            def add_const(prob):
                prob += 0 - 3 == 0
            self.assertRaises(TypeError, add_const, prob=prob)

        @gurobi_test
        def test_measuring_solving_time(self):
            print('\t Testing measuring optimization time')
            time_limit = 10
            solver_settings = dict(PULP_CBC_CMD=30, COIN_CMD=30, SCIP_CMD=30, GUROBI_CMD=50, CPLEX_CMD=50, GUROBI=50, HiGHS=50)
            bins = solver_settings.get(self.solver.name)
            if bins is None:
                return
            prob = create_bin_packing_problem(bins=bins, seed=99)
            self.solver.timeLimit = time_limit
            status = prob.solve(self.solver)
            delta = 20
            reported_time = prob.solutionTime
            if self.solver.name in ['PULP_CBC_CMD', 'COIN_CMD']:
                reported_time = prob.solutionCpuTime
            self.assertAlmostEqual(reported_time, time_limit, delta=delta, msg=f'optimization time for solver {self.solver.name}')
            self.assertTrue(prob.objective.value() is not None)
            self.assertEqual(status, const.LpStatusOptimal)
            for v in prob.variables():
                self.assertTrue(v.varValue is not None)

        @gurobi_test
        def test_time_limit_no_solution(self):
            print('\t Test time limit with no solution')
            time_limit = 1
            solver_settings = dict(HiGHS=50, PULP_CBC_CMD=30, COIN_CMD=30)
            bins = solver_settings.get(self.solver.name)
            if bins is None:
                return
            prob = create_bin_packing_problem(bins=bins, seed=99)
            self.solver.timeLimit = time_limit
            status = prob.solve(self.solver)
            self.assertEqual(prob.status, const.LpStatusNotSolved)
            self.assertEqual(status, const.LpStatusNotSolved)
            self.assertEqual(prob.sol_status, const.LpSolutionNoSolutionFound)

        def test_invalid_var_names(self):
            prob = LpProblem(self._testMethodName, const.LpMinimize)
            x = LpVariable('a')
            w = LpVariable('b')
            y = LpVariable('g', -1, 1)
            z = LpVariable('End')
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing invalid var names')
            if self.solver.name not in ['GUROBI_CMD']:
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})

        def test_LpVariable_indexs_param(self):
            """
            Test that 'indexs' param continues to work
            """
            prob = LpProblem(self._testMethodName, const.LpMinimize)
            customers = [1, 2, 3]
            agents = ['A', 'B', 'C']
            print("\t Testing 'indexs' param continues to work for LpVariable.dicts")
            assign_vars = LpVariable.dicts(name='test', indices=(customers, agents))
            for k, v in assign_vars.items():
                for a, b in v.items():
                    self.assertIsInstance(b, LpVariable)
            assign_vars = LpVariable.dicts('test', (customers, agents))
            for k, v in assign_vars.items():
                for a, b in v.items():
                    self.assertIsInstance(b, LpVariable)
            print("\t Testing 'indexs' param continues to work for LpVariable.matrix")
            assign_vars_matrix = LpVariable.matrix(name='test', indices=(customers, agents))
            for a in assign_vars_matrix:
                for b in a:
                    self.assertIsInstance(b, LpVariable)
            assign_vars_matrix = LpVariable.matrix('test', (customers, agents))
            for a in assign_vars_matrix:
                for b in a:
                    self.assertIsInstance(b, LpVariable)

        def test_LpVariable_indices_param(self):
            """
            Test that 'indices' argument works
            """
            prob = LpProblem(self._testMethodName, const.LpMinimize)
            customers = [1, 2, 3]
            agents = ['A', 'B', 'C']
            print("\t Testing 'indices' argument works in LpVariable.dicts")
            assign_vars = LpVariable.dicts(name='test', indices=(customers, agents))
            for k, v in assign_vars.items():
                for a, b in v.items():
                    self.assertIsInstance(b, LpVariable)
            print("\t Testing 'indices' param continues to work for LpVariable.matrix")
            assign_vars_matrix = LpVariable.matrix(name='test', indices=(customers, agents))
            for a in assign_vars_matrix:
                for b in a:
                    self.assertIsInstance(b, LpVariable)

        def test_parse_cplex_mipopt_solution(self):
            """
            Ensures `readsol` can parse CPLEX mipopt solutions (see issue #508).
            """
            from io import StringIO
            print('\t Testing that `readsol` can parse CPLEX mipopt solution')
            file_content = '<?xml version = "1.0" encoding="UTF-8" standalone="yes"?>\n                <CPLEXSolution version="1.2">\n                <header\n                    problemName="mipopt_solution_example.lp"\n                    solutionName="incumbent"\n                    solutionIndex="-1"\n                    objectiveValue="442"\n                    solutionTypeValue="3"\n                    solutionTypeString="primal"\n                    solutionStatusValue="101"\n                    solutionStatusString="integer optimal solution"\n                    solutionMethodString="mip"\n                    primalFeasible="1"\n                    dualFeasible="1"\n                    MIPNodes="25471"\n                    MIPIterations="282516"\n                    writeLevel="1"/>\n                <quality\n                    epInt="1.0000000000000001e-05"\n                    epRHS="9.9999999999999995e-07"\n                    maxIntInfeas="8.8817841970012523e-16"\n                    maxPrimalInfeas="0"\n                    maxX="48"\n                maxSlack="141"/>\n                <linearConstraints>\n                    <constraint name="C1" index="0" slack="0"/>\n                    <constraint name="C2" index="1" slack="0"/>\n                </linearConstraints>\n                <variables>\n                    <variable name="x" index="0" value="42"/>\n                    <variable name="y" index="1" value="0"/>\n                </variables>\n                <objectiveValues>\n                    <objective index="0" name="x" value="42"/>\n                </objectiveValues>\n                </CPLEXSolution>\n            '
            solution_file = StringIO(file_content)
            _, _, reducedCosts, shadowPrices, _, _ = CPLEX_CMD().readsol(solution_file)
            self.assertTrue(all((c is None for c in reducedCosts.values())))
            self.assertTrue(all((c is None for c in shadowPrices.values())))

        def test_options_parsing_SCIP_HIGHS(self):
            name = self._testMethodName
            prob = LpProblem(name, const.LpMinimize)
            x = LpVariable('x', 0, 4)
            y = LpVariable('y', -1, 1)
            z = LpVariable('z', 0)
            w = LpVariable('w', 0)
            prob += (x + 4 * y + 9 * z, 'obj')
            prob += (x + y <= 5, 'c1')
            prob += (x + z >= 10, 'c2')
            prob += (-y + z == 7, 'c3')
            prob += (w >= 0, 'c4')
            print('\t Testing options parsing')
            if self.solver.__class__ in [SCIP_CMD, FSCIP_CMD]:
                self.solver.options = ['limits/time', 20]
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})
            elif self.solver.__class__ in [HiGHS_CMD]:
                self.solver.options = ['time_limit', 20]
                pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})