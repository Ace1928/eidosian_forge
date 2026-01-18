import atexit
import logging
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional
import psutil
import wandb
from ..interface.interface_queue import InterfaceQueue
from ..lib import tracelog
from . import context, handler, internal_util, sender, writer
def wandb_internal(settings: 'SettingsStatic', record_q: 'Queue[Record]', result_q: 'Queue[Result]', port: Optional[int]=None, user_pid: Optional[int]=None) -> None:
    """Internal process function entrypoint.

    Read from record queue and dispatch work to various threads.

    Arguments:
        settings: settings object
        record_q: records to be handled
        result_q: for sending results back

    """
    wandb._set_internal_process()
    _setup_tracelog()
    started = time.time()
    wandb._sentry.configure_scope(process_context='internal', tags=dict(settings))

    @atexit.register
    def handle_exit(*args: 'Any') -> None:
        logger.info('Internal process exited')
    _settings = settings
    if _settings.log_internal:
        configure_logging(_settings.log_internal, _settings._log_level)
    user_pid = user_pid or os.getppid()
    pid = os.getpid()
    logger.info('W&B internal server running at pid: %s, started at: %s', pid, datetime.fromtimestamp(started))
    tracelog.annotate_queue(record_q, 'record_q')
    tracelog.annotate_queue(result_q, 'result_q')
    publish_interface = InterfaceQueue(record_q=record_q)
    stopped = threading.Event()
    threads: List[RecordLoopThread] = []
    context_keeper = context.ContextKeeper()
    send_record_q: Queue[Record] = queue.Queue()
    tracelog.annotate_queue(send_record_q, 'send_q')
    write_record_q: Queue[Record] = queue.Queue()
    tracelog.annotate_queue(write_record_q, 'write_q')
    record_sender_thread = SenderThread(settings=_settings, record_q=send_record_q, result_q=result_q, stopped=stopped, interface=publish_interface, debounce_interval_ms=5000, context_keeper=context_keeper)
    threads.append(record_sender_thread)
    record_writer_thread = WriterThread(settings=_settings, record_q=write_record_q, result_q=result_q, stopped=stopped, interface=publish_interface, sender_q=send_record_q, context_keeper=context_keeper)
    threads.append(record_writer_thread)
    record_handler_thread = HandlerThread(settings=_settings, record_q=record_q, result_q=result_q, stopped=stopped, writer_q=write_record_q, interface=publish_interface, context_keeper=context_keeper)
    threads.append(record_handler_thread)
    process_check = ProcessCheck(settings=_settings, user_pid=user_pid)
    for thread in threads:
        thread.start()
    interrupt_count = 0
    while not stopped.is_set():
        try:
            while not stopped.is_set():
                time.sleep(1)
                if process_check.is_dead():
                    logger.error('Internal process shutdown.')
                    stopped.set()
        except KeyboardInterrupt:
            interrupt_count += 1
            logger.warning(f'Internal process interrupt: {interrupt_count}')
        finally:
            if interrupt_count >= 2:
                logger.error('Internal process interrupted.')
                stopped.set()
    for thread in threads:
        thread.join()

    def close_internal_log() -> None:
        root = logging.getLogger('wandb')
        for _handler in root.handlers[:]:
            _handler.close()
            root.removeHandler(_handler)
    for thread in threads:
        exc_info = thread.get_exception()
        if exc_info:
            logger.error(f'Thread {thread.name}:', exc_info=exc_info)
            print(f'Thread {thread.name}:', file=sys.stderr)
            traceback.print_exception(*exc_info)
            wandb._sentry.exception(exc_info)
            wandb.termerror('Internal wandb error: file data was not synced')
            if not settings._disable_service:
                os._exit(-1)
            sys.exit(-1)
    close_internal_log()