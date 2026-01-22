import time
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
from _pydevd_bundle.pydevd_constants import get_thread_id
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import ObjectWrapper, wrap_attr
import pydevd_file_utils
from _pydev_bundle import pydev_log
import sys
from urllib.parse import quote
class ThreadingLogger:

    def __init__(self):
        self.start_time = cur_time()

    def set_start_time(self, time):
        self.start_time = time

    def log_event(self, frame):
        write_log = False
        self_obj = None
        if 'self' in frame.f_locals:
            self_obj = frame.f_locals['self']
            if isinstance(self_obj, threading.Thread) or self_obj.__class__ == ObjectWrapper:
                write_log = True
        if hasattr(frame, 'f_back') and frame.f_back is not None:
            back = frame.f_back
            if hasattr(back, 'f_back') and back.f_back is not None:
                back = back.f_back
                if 'self' in back.f_locals:
                    if isinstance(back.f_locals['self'], threading.Thread):
                        write_log = True
        try:
            if write_log:
                t = threadingCurrentThread()
                back = frame.f_back
                if not back:
                    return
                name, _, back_base = pydevd_file_utils.get_abs_path_real_path_and_base_from_frame(back)
                event_time = cur_time() - self.start_time
                method_name = frame.f_code.co_name
                if isinstance(self_obj, threading.Thread):
                    if not hasattr(self_obj, '_pydev_run_patched'):
                        wrap_attr(self_obj, 'run')
                    if method_name in THREAD_METHODS and (back_base not in DONT_TRACE_THREADING or (method_name in INNER_METHODS and back_base in INNER_FILES)):
                        thread_id = get_thread_id(self_obj)
                        name = self_obj.getName()
                        real_method = frame.f_code.co_name
                        parent = None
                        if real_method == '_stop':
                            if back_base in INNER_FILES and back.f_code.co_name == '_wait_for_tstate_lock':
                                back = back.f_back.f_back
                            real_method = 'stop'
                            if hasattr(self_obj, '_pydev_join_called'):
                                parent = get_thread_id(t)
                        elif real_method == 'join':
                            if not self_obj.is_alive():
                                return
                            thread_id = get_thread_id(t)
                            name = t.name
                            self_obj._pydev_join_called = True
                        if real_method == 'start':
                            parent = get_thread_id(t)
                        send_concurrency_message('threading_event', event_time, name, thread_id, 'thread', real_method, back.f_code.co_filename, back.f_lineno, back, parent=parent)
                if method_name == 'pydev_after_run_call':
                    if hasattr(frame, 'f_back') and frame.f_back is not None:
                        back = frame.f_back
                        if hasattr(back, 'f_back') and back.f_back is not None:
                            back = back.f_back
                        if 'self' in back.f_locals:
                            if isinstance(back.f_locals['self'], threading.Thread):
                                my_self_obj = frame.f_back.f_back.f_locals['self']
                                my_back = frame.f_back.f_back
                                my_thread_id = get_thread_id(my_self_obj)
                                send_massage = True
                                if hasattr(my_self_obj, '_pydev_join_called'):
                                    send_massage = False
                                if send_massage:
                                    send_concurrency_message('threading_event', event_time, 'Thread', my_thread_id, 'thread', 'stop', my_back.f_code.co_filename, my_back.f_lineno, my_back, parent=None)
                if self_obj.__class__ == ObjectWrapper:
                    if back_base in DONT_TRACE_THREADING:
                        return
                    back_back_base = pydevd_file_utils.get_abs_path_real_path_and_base_from_frame(back.f_back)[2]
                    back = back.f_back
                    if back_back_base in DONT_TRACE_THREADING:
                        return
                    if method_name == '__init__':
                        send_concurrency_message('threading_event', event_time, t.name, get_thread_id(t), 'lock', method_name, back.f_code.co_filename, back.f_lineno, back, lock_id=str(id(frame.f_locals['self'])))
                    if 'attr' in frame.f_locals and (frame.f_locals['attr'] in LOCK_METHODS or frame.f_locals['attr'] in QUEUE_METHODS):
                        real_method = frame.f_locals['attr']
                        if method_name == 'call_begin':
                            real_method += '_begin'
                        elif method_name == 'call_end':
                            real_method += '_end'
                        else:
                            return
                        if real_method == 'release_end':
                            return
                        send_concurrency_message('threading_event', event_time, t.name, get_thread_id(t), 'lock', real_method, back.f_code.co_filename, back.f_lineno, back, lock_id=str(id(self_obj)))
                        if real_method in ('put_end', 'get_end'):
                            send_concurrency_message('threading_event', event_time, t.name, get_thread_id(t), 'lock', 'release', back.f_code.co_filename, back.f_lineno, back, lock_id=str(id(self_obj)))
        except Exception:
            pydev_log.exception()