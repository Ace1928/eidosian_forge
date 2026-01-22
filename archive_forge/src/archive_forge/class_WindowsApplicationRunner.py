import os
import sys
from twisted import copyright
from twisted.application import app, internet, service
from twisted.python import log
class WindowsApplicationRunner(app.ApplicationRunner):
    """
    An ApplicationRunner which avoids unix-specific things. No
    forking, no PID files, no privileges.
    """

    def preApplication(self):
        """
        Do pre-application-creation setup.
        """
        self.oldstdout = sys.stdout
        self.oldstderr = sys.stderr
        os.chdir(self.config['rundir'])

    def postApplication(self):
        """
        Start the application and run the reactor.
        """
        service.IService(self.application).privilegedStartService()
        app.startApplication(self.application, not self.config['no_save'])
        app.startApplication(internet.TimerService(0.1, lambda: None), 0)
        self.startReactor(None, self.oldstdout, self.oldstderr)
        log.msg('Server Shut Down.')