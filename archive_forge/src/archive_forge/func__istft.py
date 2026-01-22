import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_concat_from_sequence import _concat_from_sequence
from onnx.reference.ops.op_dft import _cfft as _dft
from onnx.reference.ops.op_slice import _slice
def _istft(x, fft_length: int, hop_length, window, onesided=False):
    """Reverses of `stft`."""
    zero = [0]
    one = [1]
    two = [2]
    axisf = [-2]
    n_frames = x.shape[-2]
    expected_signal_len = fft_length + hop_length * (n_frames - 1)
    seqr = []
    seqi = []
    seqc = []
    for fs in range(n_frames):
        begin = fs
        end = fs + 1
        frame_x = np.squeeze(_slice(x, np.array([begin]), np.array([end]), axisf), axis=axisf[0])
        ift = _dft(frame_x, fft_length, axis=-1, onesided=onesided, normalize=True)
        n_dims = len(ift.shape)
        n_dims_1 = n_dims - 1
        sliced = _slice(ift, np.array(zero), np.array(one), [n_dims_1])
        ytmp = np.squeeze(sliced, axis=n_dims_1)
        ctmp = np.full(ytmp.shape, fill_value=1, dtype=x.dtype) * window
        shape_begin = ytmp.shape[:-1]
        n_left = fs * hop_length
        size = ytmp.shape[-1]
        n_right = expected_signal_len - (n_left + size)
        left_shape = (*shape_begin, n_left)
        right_shape = (*shape_begin, n_right)
        right = np.zeros(right_shape, dtype=x.dtype)
        left = np.zeros(left_shape, dtype=x.dtype)
        y = _concat(left, ytmp, right, axis=-1)
        yc = _concat(left, ctmp, right, axis=-1)
        sliced = _slice(ift, np.array(one), np.array(two), [n_dims_1])
        itmp = np.squeeze(sliced, axis=n_dims_1)
        yi = _concat(left, itmp, right, axis=-1)
        seqr.append(_unsqueeze(y, axis=-1))
        seqi.append(_unsqueeze(yi, axis=-1))
        seqc.append(_unsqueeze(yc, axis=-1))
    redr = _concat_from_sequence(seqr, axis=-1, new_axis=0)
    redi = _concat_from_sequence(seqi, axis=-1, new_axis=0)
    redc = _concat_from_sequence(seqc, axis=-1, new_axis=0)
    resr = redr.sum(axis=-1, keepdims=0)
    resi = redi.sum(axis=-1, keepdims=0)
    resc = redc.sum(axis=-1, keepdims=0)
    rr = resr / resc
    ri = resi / resc
    rr0 = np.expand_dims(rr, axis=0)
    ri0 = np.expand_dims(ri, axis=0)
    conc = _concat(rr0, ri0, axis=0)
    result_shape = conc.shape
    reshaped_result = conc.reshape((2, -1))
    transposed = np.transpose(reshaped_result, (1, 0))
    other_dimensions = result_shape[1:]
    final_shape = _concat(other_dimensions, two, axis=0)
    final = transposed.reshape(final_shape)
    return final