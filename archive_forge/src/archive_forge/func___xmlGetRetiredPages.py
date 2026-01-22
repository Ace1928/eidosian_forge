from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __xmlGetRetiredPages(handle, filter):
    retiredPages = ''
    includeRetiredPages = False
    causes = ['multiple_single_bit_retirement', 'double_bit_retirement']
    for idx in range(NVML_PAGE_RETIREMENT_CAUSE_COUNT):
        if NVSMI_ALL in filter or (NVSMI_RETIREDPAGES_SINGLE_BIT_ECC_COUNT in filter and idx == 0) or (NVSMI_RETIREDPAGES_DOUBLE_BIT_ECC_COUNT in filter and idx == 1):
            retiredPages += '      <' + causes[idx] + '>\n'
            retiredPages += nvidia_smi.__xmlGetRetiredPagesByCause(handle, idx)
            retiredPages += '      </' + causes[idx] + '>\n'
            includeRetiredPages = True
    if NVSMI_ALL in filter or NVSMI_RETIREDPAGES_PENDING in filter:
        retiredPages += '      <pending_retirement>'
        try:
            if NVML_FEATURE_DISABLED == nvmlDeviceGetRetiredPagesPendingStatus(handle):
                retiredPages += 'No'
            else:
                retiredPages += 'Yes'
        except NVMLError as err:
            retiredPages += nvidia_smi.__handleError(err)
        retiredPages += '</pending_retirement>\n'
        includeRetiredPages = True
    return (retiredPages if len(retiredPages) > 0 else None, includeRetiredPages)