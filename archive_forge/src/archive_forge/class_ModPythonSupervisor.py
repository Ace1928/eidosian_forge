import os
import re
import cherrypy
from cherrypy.test import helper
class ModPythonSupervisor(helper.Supervisor):
    using_apache = True
    using_wsgi = False
    template = None

    def __str__(self):
        return 'ModPython Server on %s:%s' % (self.host, self.port)

    def start(self, modulename):
        mpconf = CONF_PATH
        if not os.path.isabs(mpconf):
            mpconf = os.path.join(curdir, mpconf)
        with open(mpconf, 'wb') as f:
            f.write(self.template % {'port': self.port, 'modulename': modulename, 'host': self.host})
        result = read_process(APACHE_PATH, '-k start -f %s' % mpconf)
        if result:
            print(result)

    def stop(self):
        """Gracefully shutdown a server that is serving forever."""
        read_process(APACHE_PATH, '-k stop')