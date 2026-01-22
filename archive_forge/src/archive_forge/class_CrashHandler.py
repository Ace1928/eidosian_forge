import sys
import traceback
from pprint import pformat
from pathlib import Path
from IPython.core import ultratb
from IPython.core.release import author_email
from IPython.utils.sysinfo import sys_info
from IPython.utils.py3compat import input
from IPython.core.release import __version__ as version
from typing import Optional
class CrashHandler(object):
    """Customizable crash handlers for IPython applications.

    Instances of this class provide a :meth:`__call__` method which can be
    used as a ``sys.excepthook``.  The :meth:`__call__` signature is::

        def __call__(self, etype, evalue, etb)
    """
    message_template = _default_message_template
    section_sep = '\n\n' + '*' * 75 + '\n\n'

    def __init__(self, app, contact_name: Optional[str]=None, contact_email: Optional[str]=None, bug_tracker: Optional[str]=None, show_crash_traceback: bool=True, call_pdb: bool=False):
        """Create a new crash handler

        Parameters
        ----------
        app : Application
            A running :class:`Application` instance, which will be queried at
            crash time for internal information.
        contact_name : str
            A string with the name of the person to contact.
        contact_email : str
            A string with the email address of the contact.
        bug_tracker : str
            A string with the URL for your project's bug tracker.
        show_crash_traceback : bool
            If false, don't print the crash traceback on stderr, only generate
            the on-disk report
        call_pdb
            Whether to call pdb on crash

        Attributes
        ----------
        These instances contain some non-argument attributes which allow for
        further customization of the crash handler's behavior. Please see the
        source for further details.

        """
        self.crash_report_fname = 'Crash_report_%s.txt' % app.name
        self.app = app
        self.call_pdb = call_pdb
        self.show_crash_traceback = show_crash_traceback
        self.info = dict(app_name=app.name, contact_name=contact_name, contact_email=contact_email, bug_tracker=bug_tracker, crash_report_fname=self.crash_report_fname)

    def __call__(self, etype, evalue, etb):
        """Handle an exception, call for compatible with sys.excepthook"""
        sys.excepthook = sys.__excepthook__
        color_scheme = 'NoColor'
        try:
            rptdir = self.app.ipython_dir
        except:
            rptdir = Path.cwd()
        if rptdir is None or not Path.is_dir(rptdir):
            rptdir = Path.cwd()
        report_name = rptdir / self.crash_report_fname
        self.crash_report_fname = report_name
        self.info['crash_report_fname'] = report_name
        TBhandler = ultratb.VerboseTB(color_scheme=color_scheme, long_header=1, call_pdb=self.call_pdb)
        if self.call_pdb:
            TBhandler(etype, evalue, etb)
            return
        else:
            traceback = TBhandler.text(etype, evalue, etb, context=31)
        if self.show_crash_traceback:
            print(traceback, file=sys.stderr)
        try:
            report = open(report_name, 'w', encoding='utf-8')
        except:
            print('Could not create crash report on disk.', file=sys.stderr)
            return
        with report:
            print('\n' + '*' * 70 + '\n', file=sys.stderr)
            print(self.message_template.format(**self.info), file=sys.stderr)
            report.write(self.make_report(traceback))
        input('Hit <Enter> to quit (your terminal may close):')

    def make_report(self, traceback):
        """Return a string containing a crash report."""
        sec_sep = self.section_sep
        report = ['*' * 75 + '\n\n' + 'IPython post-mortem report\n\n']
        rpt_add = report.append
        rpt_add(sys_info())
        try:
            config = pformat(self.app.config)
            rpt_add(sec_sep)
            rpt_add('Application name: %s\n\n' % self.app_name)
            rpt_add('Current user configuration structure:\n\n')
            rpt_add(config)
        except:
            pass
        rpt_add(sec_sep + 'Crash traceback:\n\n' + traceback)
        return ''.join(report)