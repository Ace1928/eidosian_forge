import logging
import os
import platform as _platform
import sys
from datetime import datetime
from functools import partial
from billiard.common import REMAP_SIGTERM
from billiard.process import current_process
from kombu.utils.encoding import safe_str
from celery import VERSION_BANNER, platforms, signals
from celery.app import trace
from celery.loaders.app import AppLoader
from celery.platforms import EX_FAILURE, EX_OK, check_privileges
from celery.utils import static, term
from celery.utils.debug import cry
from celery.utils.imports import qualname
from celery.utils.log import get_logger, in_sighandler, set_in_sighandler
from celery.utils.text import pluralize
from celery.worker import WorkController
def _shutdown_handler(worker, sig='TERM', how='Warm', callback=None, exitcode=EX_OK):

    def _handle_request(*args):
        with in_sighandler():
            from celery.worker import state
            if current_process()._name == 'MainProcess':
                if callback:
                    callback(worker)
                safe_say(f'worker: {how} shutdown (MainProcess)', sys.__stdout__)
                signals.worker_shutting_down.send(sender=worker.hostname, sig=sig, how=how, exitcode=exitcode)
            setattr(state, {'Warm': 'should_stop', 'Cold': 'should_terminate'}[how], exitcode)
    _handle_request.__name__ = str(f'worker_{how}')
    platforms.signals[sig] = _handle_request