import numpy as np
from scipy import signal
 multivariate linear filter

    x (TxK)
    B (PxKxK)

    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) } +
                sum{_q}sum{_k} { e(t-Q:t,:) .* C(:,:,i) }for all i = 0,K-1

    