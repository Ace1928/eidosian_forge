import collections
import queue
import torch
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper
def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    torch.set_num_threads(1)
    if device == 'cuda':
        torch.cuda.set_device(device_id)
    elif device == 'xpu':
        torch.xpu.set_device(device_id)
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

    def do_one_step():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and (not isinstance(data, ExceptionWrapper)):
            try:
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(where=f'in pin memory thread for device {device_id}')
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
    while not done_event.is_set():
        do_one_step()