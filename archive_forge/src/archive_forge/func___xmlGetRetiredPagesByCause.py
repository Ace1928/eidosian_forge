from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __xmlGetRetiredPagesByCause(handle, cause):
    retiredPagedByCause = ''
    error = None
    count = 0
    try:
        pages = nvmlDeviceGetRetiredPages(handle, cause)
        count = sum(map(len, pages))
    except NVMLError as err:
        error = nvidia_smi.__handleError(err)
        pages = None
    retiredPagedByCause += '        <retired_count>' + nvidia_smi.__toString(count) + '</retired_count>\n'
    if pages is not None:
        retiredPagedByCause += '        <retired_page_addresses>\n'
        for page in pages:
            retiredPagedByCause += '          <retired_page_address>' + '0x%016x' % page + '</retired_page_address>\n'
        retiredPagedByCause += '        </retired_page_addresses>\n'
    else:
        retiredPagedByCause += '        <retired_page_addresses>' + error + '</retired_page_addresses>\n'
    return retiredPagedByCause if count > 0 else ''