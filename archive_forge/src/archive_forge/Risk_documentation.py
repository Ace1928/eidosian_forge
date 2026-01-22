import math


    The formulation here is from Eqns 4.22 and 4.23 on pg 108 of
    Cherkassky and Mulier's book "Learning From Data" Wiley, 1998.

    **Arguments**

      - VCDim: the VC dimension of the system

      - nData: the number of data points used

      - nWrong: the number of data points misclassified

      - conf: the confidence to be used for this risk bound

      - a1, a2: constants in the risk equation. Restrictions on these values:

          - 0 <= a1 <= 4

          - 0 <= a2 <= 2

    **Returns**

      - a float


    **Notes**

     - This appears to behave reasonably

     - the equality a1=1.0 is by analogy to Burges's paper.

  