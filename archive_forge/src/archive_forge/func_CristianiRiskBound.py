import math
def CristianiRiskBound(VCDim, nData, nWrong, conf):
    """
    the formulation here is from pg 58, Theorem 4.6 of the book
    "An Introduction to Support Vector Machines" by Cristiani and Shawe-Taylor
    Cambridge University Press, 2000


    **Arguments**

      - VCDim: the VC dimension of the system

      - nData: the number of data points used

      - nWrong: the number of data points misclassified

      - conf: the confidence to be used for this risk bound


    **Returns**

      - a float

    **Notes**

      - this generates odd (mismatching) values

  """
    d = VCDim
    delta = conf
    l = nData
    k = nWrong
    structRisk = math.sqrt(4.0 / nData * (d * log2(2.0 * math.e * l / d) + log2(4.0 / delta)))
    rEmp = 2.0 * k / l
    return rEmp + structRisk