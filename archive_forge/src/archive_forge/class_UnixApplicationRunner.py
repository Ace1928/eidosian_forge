import errno
import os
import pwd
import sys
import traceback
from twisted import copyright, logger
from twisted.application import app, service
from twisted.internet.interfaces import IReactorDaemonize
from twisted.python import log, logfile, usage
from twisted.python.runtime import platformType
from twisted.python.util import gidFromString, switchUID, uidFromString, untilConcludes
class UnixApplicationRunner(app.ApplicationRunner):
    """
    An ApplicationRunner which does Unix-specific things, like fork,
    shed privileges, and maintain a PID file.
    """
    loggerFactory = UnixAppLogger

    def preApplication(self):
        """
        Do pre-application-creation setup.
        """
        checkPID(self.config['pidfile'])
        self.config['nodaemon'] = self.config['nodaemon'] or self.config['debug']
        self.oldstdout = sys.stdout
        self.oldstderr = sys.stderr

    def _formatChildException(self, exception):
        """
        Format the C{exception} in preparation for writing to the
        status pipe.  This does the right thing on Python 2 if the
        exception's message is Unicode, and in all cases limits the
        length of the message afte* encoding to 100 bytes.

        This means the returned message may be truncated in the middle
        of a unicode escape.

        @type exception: L{Exception}
        @param exception: The exception to format.

        @return: The formatted message, suitable for writing to the
            status pipe.
        @rtype: L{bytes}
        """
        exceptionLine = traceback.format_exception_only(exception.__class__, exception)[-1]
        formattedMessage = f'1 {exceptionLine.strip()}'
        formattedMessage = formattedMessage.encode('ascii', 'backslashreplace')
        return formattedMessage[:100]

    def postApplication(self):
        """
        To be called after the application is created: start the application
        and run the reactor. After the reactor stops, clean up PID files and
        such.
        """
        try:
            self.startApplication(self.application)
        except Exception as ex:
            statusPipe = self.config.get('statusPipe', None)
            if statusPipe is not None:
                message = self._formatChildException(ex)
                untilConcludes(os.write, statusPipe, message)
                untilConcludes(os.close, statusPipe)
            self.removePID(self.config['pidfile'])
            raise
        else:
            statusPipe = self.config.get('statusPipe', None)
            if statusPipe is not None:
                untilConcludes(os.write, statusPipe, b'0')
                untilConcludes(os.close, statusPipe)
        self.startReactor(None, self.oldstdout, self.oldstderr)
        self.removePID(self.config['pidfile'])

    def removePID(self, pidfile):
        """
        Remove the specified PID file, if possible.  Errors are logged, not
        raised.

        @type pidfile: C{str}
        @param pidfile: The path to the PID tracking file.
        """
        if not pidfile:
            return
        try:
            os.unlink(pidfile)
        except OSError as e:
            if e.errno == errno.EACCES or e.errno == errno.EPERM:
                log.msg('Warning: No permission to delete pid file')
            else:
                log.err(e, 'Failed to unlink PID file:')
        except BaseException:
            log.err(None, 'Failed to unlink PID file:')

    def setupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
        """
        Set the filesystem root, the working directory, and daemonize.

        @type chroot: C{str} or L{None}
        @param chroot: If not None, a path to use as the filesystem root (using
            L{os.chroot}).

        @type rundir: C{str}
        @param rundir: The path to set as the working directory.

        @type nodaemon: C{bool}
        @param nodaemon: A flag which, if set, indicates that daemonization
            should not be done.

        @type umask: C{int} or L{None}
        @param umask: The value to which to change the process umask.

        @type pidfile: C{str} or L{None}
        @param pidfile: If not L{None}, the path to a file into which to put
            the PID of this process.
        """
        daemon = not nodaemon
        if chroot is not None:
            os.chroot(chroot)
            if rundir == '.':
                rundir = '/'
        os.chdir(rundir)
        if daemon and umask is None:
            umask = 63
        if umask is not None:
            os.umask(umask)
        if daemon:
            from twisted.internet import reactor
            self.config['statusPipe'] = self.daemonize(reactor)
        if pidfile:
            with open(pidfile, 'wb') as f:
                f.write(b'%d' % (os.getpid(),))

    def daemonize(self, reactor):
        """
        Daemonizes the application on Unix. This is done by the usual double
        forking approach.

        @see: U{http://code.activestate.com/recipes/278731/}
        @see: W. Richard Stevens,
            "Advanced Programming in the Unix Environment",
            1992, Addison-Wesley, ISBN 0-201-56317-7

        @param reactor: The reactor in use.  If it provides
            L{IReactorDaemonize}, its daemonization-related callbacks will be
            invoked.

        @return: A writable pipe to be used to report errors.
        @rtype: C{int}
        """
        if IReactorDaemonize.providedBy(reactor):
            reactor.beforeDaemonize()
        r, w = os.pipe()
        if os.fork():
            code = self._waitForStart(r)
            os.close(r)
            os._exit(code)
        os.setsid()
        if os.fork():
            os._exit(0)
        null = os.open('/dev/null', os.O_RDWR)
        for i in range(3):
            try:
                os.dup2(null, i)
            except OSError as e:
                if e.errno != errno.EBADF:
                    raise
        os.close(null)
        if IReactorDaemonize.providedBy(reactor):
            reactor.afterDaemonize()
        return w

    def _waitForStart(self, readPipe: int) -> int:
        """
        Wait for the daemonization success.

        @param readPipe: file descriptor to read start information from.
        @type readPipe: C{int}

        @return: code to be passed to C{os._exit}: 0 for success, 1 for error.
        @rtype: C{int}
        """
        data = untilConcludes(os.read, readPipe, 100)
        dataRepr = repr(data[2:])
        if data != b'0':
            msg = 'An error has occurred: {}\nPlease look at log file for more information.\n'.format(dataRepr)
            untilConcludes(sys.__stderr__.write, msg)
            return 1
        return 0

    def shedPrivileges(self, euid, uid, gid):
        """
        Change the UID and GID or the EUID and EGID of this process.

        @type euid: C{bool}
        @param euid: A flag which, if set, indicates that only the I{effective}
            UID and GID should be set.

        @type uid: C{int} or L{None}
        @param uid: If not L{None}, the UID to which to switch.

        @type gid: C{int} or L{None}
        @param gid: If not L{None}, the GID to which to switch.
        """
        if uid is not None or gid is not None:
            extra = euid and 'e' or ''
            desc = f'{extra}uid/{extra}gid {uid}/{gid}'
            try:
                switchUID(uid, gid, euid)
            except OSError as e:
                log.msg('failed to set {}: {} (are you root?) -- exiting.'.format(desc, e))
                sys.exit(1)
            else:
                log.msg(f'set {desc}')

    def startApplication(self, application):
        """
        Configure global process state based on the given application and run
        the application.

        @param application: An object which can be adapted to
            L{service.IProcess} and L{service.IService}.
        """
        process = service.IProcess(application)
        if not self.config['originalname']:
            launchWithName(process.processName)
        self.setupEnvironment(self.config['chroot'], self.config['rundir'], self.config['nodaemon'], self.config['umask'], self.config['pidfile'])
        service.IService(application).privilegedStartService()
        uid, gid = (self.config['uid'], self.config['gid'])
        if uid is None:
            uid = process.uid
        if gid is None:
            gid = process.gid
        if uid is not None and gid is None:
            gid = pwd.getpwuid(uid).pw_gid
        self.shedPrivileges(self.config['euid'], uid, gid)
        app.startApplication(application, not self.config['no_save'])