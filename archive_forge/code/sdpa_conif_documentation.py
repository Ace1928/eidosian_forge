import cvxpy.settings as s
from cvxpy.constraints import PSD, NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver import Solver

        CVXPY represents cone programs as
            (P) min_x { c^T x : A x + b \in K } + d

        SDPA Python takes a conic program in CLP format:
            (P) min_x { c^T x : A x - b \in J, x \in K }

        CVXPY (P) -> CLP (P), by
            - flipping sign of b
            - setting J of CLP (P) to K of CVXPY (P)
            - setting K of CLP (P) to a free cone

        CLP format is a generalization of the SeDuMi format. Both formats are explained at
        https://sdpa-python.github.io/docs/formats/

        Internally, SDPA Python will reduce CLP form to SeDuMi dual form using `clp_toLMI`.
        In SeDuMi format, the dual is in LMI form. In SDPA format, the primal is in LMI form.
        The backend (i.e. `libsdpa.a` or `libsdpa_gmp.a`) uses the SDPA format.

        For details on the reverse relationship between SDPA and SeDuMi formats, please see
        https://sdpa-python.github.io/docs/formats/sdpa_sedumi.html
        