from typing import Any, Dict, Sequence, Union
import cmath
import math
import cirq
from cirq import protocols
from cirq._doc import document
import numpy as np
The Mølmer–Sørensen (MS) gate is a two qubit gate native to trapped ions.

    The unitary matrix of this gate for parameters $\phi_0$, $\phi_1$ and $\theta$ is

    $$
    \begin{bmatrix}
        \cos\frac{\theta}{2} & 0 & 0 & -ie^{-i2\pi(\phi_0+\phi_1)}\sin\frac{\theta}{2} \\
        0 & \cos\frac{\theta}{2} & -ie^{-i2\pi(\phi_0-\phi_1)}\sin\frac{\theta}{2} & 0 \\
        0 & -ie^{i2\pi(\phi_0-\phi_1)}\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
        -ie^{i2\pi(\phi_0+\phi_1)}\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
    \end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    