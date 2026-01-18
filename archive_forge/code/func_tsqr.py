import numpy as np
import ray
import ray.experimental.array.remote as ra
from . import core
@ray.remote(num_returns=2)
def tsqr(a):
    """Perform a QR decomposition of a tall-skinny matrix.

    Args:
        a: A distributed matrix with shape MxN (suppose K = min(M, N)).

    Returns:
        A tuple of q (a DistArray) and r (a numpy array) satisfying the
            following.
            - If q_full = ray.get(DistArray, q).assemble(), then
              q_full.shape == (M, K).
            - np.allclose(np.dot(q_full.T, q_full), np.eye(K)) == True.
            - If r_val = ray.get(np.ndarray, r), then r_val.shape == (K, N).
            - np.allclose(r, np.triu(r)) == True.
    """
    if len(a.shape) != 2:
        raise Exception('tsqr requires len(a.shape) == 2, but a.shape is {}'.format(a.shape))
    if a.num_blocks[1] != 1:
        raise Exception('tsqr requires a.num_blocks[1] == 1, but a.num_blocks is {}'.format(a.num_blocks))
    num_blocks = a.num_blocks[0]
    K = int(np.ceil(np.log2(num_blocks))) + 1
    q_tree = np.empty((num_blocks, K), dtype=object)
    current_rs = []
    for i in range(num_blocks):
        block = a.object_refs[i, 0]
        q, r = ra.linalg.qr.remote(block)
        q_tree[i, 0] = q
        current_rs.append(r)
    for j in range(1, K):
        new_rs = []
        for i in range(int(np.ceil(1.0 * len(current_rs) / 2))):
            stacked_rs = ra.vstack.remote(*current_rs[2 * i:2 * i + 2])
            q, r = ra.linalg.qr.remote(stacked_rs)
            q_tree[i, j] = q
            new_rs.append(r)
        current_rs = new_rs
    assert len(current_rs) == 1, 'len(current_rs) = ' + str(len(current_rs))
    if a.shape[0] >= a.shape[1]:
        q_shape = a.shape
    else:
        q_shape = [a.shape[0], a.shape[0]]
    q_num_blocks = core.DistArray.compute_num_blocks(q_shape)
    q_object_refs = np.empty(q_num_blocks, dtype=object)
    q_result = core.DistArray(q_shape, q_object_refs)
    for i in range(num_blocks):
        q_block_current = q_tree[i, 0]
        ith_index = i
        for j in range(1, K):
            if np.mod(ith_index, 2) == 0:
                lower = [0, 0]
                upper = [a.shape[1], core.BLOCK_SIZE]
            else:
                lower = [a.shape[1], 0]
                upper = [2 * a.shape[1], core.BLOCK_SIZE]
            ith_index //= 2
            q_block_current = ra.dot.remote(q_block_current, ra.subarray.remote(q_tree[ith_index, j], lower, upper))
        q_result.object_refs[i] = q_block_current
    r = current_rs[0]
    return (q_result, ray.get(r))