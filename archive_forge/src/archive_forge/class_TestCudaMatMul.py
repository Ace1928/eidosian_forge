import numpy as np
from numba import cuda, float32, void
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import config
class TestCudaMatMul(CUDATestCase):

    def test_func(self):

        @cuda.jit(void(float32[:, ::1], float32[:, ::1], float32[:, ::1]))
        def cu_square_matrix_mul(A, B, C):
            sA = cuda.shared.array(shape=SM_SIZE, dtype=float32)
            sB = cuda.shared.array(shape=(tpb, tpb), dtype=float32)
            tx = cuda.threadIdx.x
            ty = cuda.threadIdx.y
            bx = cuda.blockIdx.x
            by = cuda.blockIdx.y
            bw = cuda.blockDim.x
            bh = cuda.blockDim.y
            x = tx + bx * bw
            y = ty + by * bh
            acc = float32(0)
            for i in range(bpg):
                if x < n and y < n:
                    sA[ty, tx] = A[y, tx + i * tpb]
                    sB[ty, tx] = B[ty + i * tpb, x]
                cuda.syncthreads()
                if x < n and y < n:
                    for j in range(tpb):
                        acc += sA[ty, j] * sB[j, tx]
                cuda.syncthreads()
            if x < n and y < n:
                C[y, x] = acc
        np.random.seed(42)
        A = np.array(np.random.random((n, n)), dtype=np.float32)
        B = np.array(np.random.random((n, n)), dtype=np.float32)
        C = np.empty_like(A)
        stream = cuda.stream()
        with stream.auto_synchronize():
            dA = cuda.to_device(A, stream)
            dB = cuda.to_device(B, stream)
            dC = cuda.to_device(C, stream)
            cu_square_matrix_mul[(bpg, bpg), (tpb, tpb), stream](dA, dB, dC)
            dC.copy_to_host(C, stream)
        Cans = np.dot(A, B)
        np.testing.assert_allclose(C, Cans, rtol=1e-05)