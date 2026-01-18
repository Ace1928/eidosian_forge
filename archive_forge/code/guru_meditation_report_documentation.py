import inspect
import logging
import os
import signal
import stat
import sys
import threading
import time
import traceback
from oslo_utils import timeutils
from oslo_reports.generators import conf as cgen
from oslo_reports.generators import process as prgen
from oslo_reports.generators import threading as tgen
from oslo_reports.generators import version as pgen
from oslo_reports import report
The Signal Handler

        This method (indirectly) handles receiving a registered signal and
        dumping the Guru Meditation Report to stderr or a file in a given dir.
        If service name and log dir are not None, the report will be dumped to
        a file named $service_name_gurumeditation_$current_time in the log_dir
        directory.
        This method is designed to be curried into a proper signal handler by
        currying out the version
        parameter.

        :param version: the version object for the current product
        :param service_name: this program name used to construct logfile name
        :param logdir: path to a log directory where to create a file
        :param frame: the frame object provided to the signal handler
        