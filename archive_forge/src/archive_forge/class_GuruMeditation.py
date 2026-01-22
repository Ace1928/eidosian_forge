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
class GuruMeditation(object):
    """A Guru Meditation Report Mixin/Base Class

    This class is a base class for Guru Meditation Reports.
    It provides facilities for registering sections and
    setting up functionality to auto-run the report on
    a certain signal or use file modification events.

    This class should always be used in conjunction with
    a Report class via multiple inheritance.  It should
    always come first in the class list to ensure the
    MRO is correct.
    """
    timestamp_fmt = '%Y%m%d%H%M%S'

    def __init__(self, version_obj, sig_handler_tb=None, *args, **kwargs):
        self.version_obj = version_obj
        self.traceback = sig_handler_tb
        super(GuruMeditation, self).__init__(*args, **kwargs)
        self.start_section_index = len(self.sections)

    @classmethod
    def register_section(cls, section_title, generator):
        """Register a New Section

        This method registers a persistent section for the current
        class.

        :param str section_title: the title of the section
        :param generator: the generator for the section
        """
        try:
            cls.persistent_sections.append([section_title, generator])
        except AttributeError:
            cls.persistent_sections = [[section_title, generator]]

    @classmethod
    def setup_autorun(cls, version, service_name=None, log_dir=None, signum=None, conf=None, setup_signal=True):
        """Set Up Auto-Run

        This method sets up the Guru Meditation Report to automatically
        get dumped to stderr or a file in a given dir when the given signal
        is received. It can also use file modification events instead of
        signals.

        :param version: the version object for the current product
        :param service_name: this program name used to construct logfile name
        :param logdir: path to a log directory where to create a file
        :param signum: the signal to associate with running the report
        :param conf: Configuration object, managed by the caller.
        :param setup_signal: Set up a signal handler
        """
        if log_dir is None and conf is not None:
            log_dir = conf.oslo_reports.log_dir
        if signum:
            if setup_signal:
                cls._setup_signal(signum, version, service_name, log_dir)
            return
        if conf and conf.oslo_reports.file_event_handler:
            cls._setup_file_watcher(conf.oslo_reports.file_event_handler, conf.oslo_reports.file_event_handler_interval, version, service_name, log_dir)
        elif setup_signal and hasattr(signal, 'SIGUSR2'):
            cls._setup_signal(signal.SIGUSR2, version, service_name, log_dir)

    @classmethod
    def _setup_file_watcher(cls, filepath, interval, version, service_name, log_dir):
        st = os.stat(filepath)
        if not bool(st.st_mode & stat.S_IRGRP):
            LOG.error("Guru Meditation Report does not have read permissions to '%s' file.", filepath)

        def _handler():
            mtime = time.time()
            while True:
                try:
                    stat = os.stat(filepath)
                    if stat.st_mtime > mtime:
                        cls.handle_signal(version, service_name, log_dir, None)
                        mtime = stat.st_mtime
                except OSError:
                    msg = 'Guru Meditation Report cannot read ' + "'{0}' file".format(filepath)
                    raise IOError(msg)
                finally:
                    time.sleep(interval)
        th = threading.Thread(target=_handler)
        th.daemon = True
        th.start()

    @classmethod
    def _setup_signal(cls, signum, version, service_name, log_dir):
        signal.signal(signum, lambda sn, f: cls.handle_signal(version, service_name, log_dir, f))

    @classmethod
    def handle_signal(cls, version, service_name, log_dir, frame):
        """The Signal Handler

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
        """
        try:
            res = cls(version, frame).run()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print('Unable to run Guru Meditation Report!', file=sys.stderr)
        else:
            if log_dir:
                service_name = service_name or os.path.basename(inspect.stack()[-1][1])
                filename = '%s_gurumeditation_%s' % (service_name, timeutils.utcnow().strftime(cls.timestamp_fmt))
                filepath = os.path.join(log_dir, filename)
                try:
                    with open(filepath, 'w') as dumpfile:
                        dumpfile.write(res)
                except Exception:
                    print('Unable to dump Guru Meditation Report to file %s' % (filepath,), file=sys.stderr)
            else:
                print(res, file=sys.stderr)

    def _readd_sections(self):
        del self.sections[self.start_section_index:]
        self.add_section('Package', pgen.PackageReportGenerator(self.version_obj))
        self.add_section('Threads', tgen.ThreadReportGenerator(self.traceback))
        if greenlet:
            self.add_section('Green Threads', tgen.GreenThreadReportGenerator())
        self.add_section('Processes', prgen.ProcessReportGenerator())
        self.add_section('Configuration', cgen.ConfigReportGenerator())
        try:
            for section_title, generator in self.persistent_sections:
                self.add_section(section_title, generator)
        except AttributeError:
            pass

    def run(self):
        self._readd_sections()
        return super(GuruMeditation, self).run()