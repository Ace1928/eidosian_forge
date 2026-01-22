import numpy as np

        Parameters
        ----------
        stp : float
            The current estimate of a satisfactory step. On initial entry, a
            positive initial estimate must be provided.
        f : float
            On first call f is the value of the function at 0. On subsequent
            entries f should be the value of the function at stp.
        g : float
            On initial entry g is the derivative of the function at 0. On
            subsequent entries g is the derivative of the function at stp.
        task : bytes
            On initial entry task must be set to 'START'.

        On exit with convergence, a warning or an error, the
           variable task contains additional information.


        Returns
        -------
        stp, f, g, task: tuple

            stp : float
                the current estimate of a satisfactory step if task = 'FG'. If
                task = 'CONV' then stp satisfies the sufficient decrease and
                curvature condition.
            f : float
                the value of the function at stp.
            g : float
                the derivative of the function at stp.
            task : bytes
                On exit task indicates the required action:

               If task(1:2) == b'FG' then evaluate the function and
               derivative at stp and call dcsrch again.

               If task(1:4) == b'CONV' then the search is successful.

               If task(1:4) == b'WARN' then the subroutine is not able
               to satisfy the convergence conditions. The exit value of
               stp contains the best point found during the search.

              If task(1:5) == b'ERROR' then there is an error in the
              input arguments.
        