from math import ceil
import cupy
def _get_tpb_bpg():
    device_id = cupy.cuda.Device()
    numSM = device_id.attributes['MultiProcessorCount']
    threadsperblock = 512
    blockspergrid = numSM * 20
    return (threadsperblock, blockspergrid)