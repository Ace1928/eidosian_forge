import numpy as np
import scipy.linalg
def _diff_pade7(A, E, ident):
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    U = A.dot(b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = A.dot(b[7] * M6 + b[5] * M4 + b[3] * M2) + E.dot(b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    Lv = b[6] * M6 + b[4] * M4 + b[2] * M2
    return (U, V, Lu, Lv)