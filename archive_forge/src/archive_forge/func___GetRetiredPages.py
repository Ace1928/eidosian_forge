from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __GetRetiredPages(handle, filter):
    retiredPages = {}
    includeRetiredPages = False
    causes = ['multiple_single_bit_retirement', 'double_bit_retirement']
    for idx in range(NVML_PAGE_RETIREMENT_CAUSE_COUNT):
        if NVSMI_ALL in filter or (NVSMI_RETIREDPAGES_SINGLE_BIT_ECC_COUNT in filter and idx == 0) or (NVSMI_RETIREDPAGES_DOUBLE_BIT_ECC_COUNT in filter and idx == 1):
            retiredPages[causes[idx]] = nvidia_smi.__GetRetiredPagesByCause(handle, idx)
            includeRetiredPages = True
    if NVSMI_ALL in filter or NVSMI_RETIREDPAGES_PENDING in filter:
        pending_retirement = ''
        try:
            if NVML_FEATURE_DISABLED == nvmlDeviceGetRetiredPagesPendingStatus(handle):
                pending_retirement = 'No'
            else:
                pending_retirement = 'Yes'
        except NVMLError as err:
            pending_retirement = nvidia_smi.__handleError(err)
        retiredPages['pending_retirement'] = pending_retirement
        includeRetiredPages = True
    return (retiredPages if len(retiredPages.values()) > 0 else None, includeRetiredPages)