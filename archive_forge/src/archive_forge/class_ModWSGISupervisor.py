import os
import re
import sys
import time
import portend
from cheroot.test import webtest
import cherrypy
from cherrypy.test import helper
class ModWSGISupervisor(helper.Supervisor):
    """Server Controller for ModWSGI and CherryPy."""
    using_apache = True
    using_wsgi = True
    template = conf_modwsgi

    def __str__(self):
        return 'ModWSGI Server on %s:%s' % (self.host, self.port)

    def start(self, modulename):
        mpconf = CONF_PATH
        if not os.path.isabs(mpconf):
            mpconf = os.path.join(curdir, mpconf)
        with open(mpconf, 'wb') as f:
            output = self.template % {'port': self.port, 'testmod': modulename, 'curdir': curdir}
            f.write(output)
        result = read_process(APACHE_PATH, '-k start -f %s' % mpconf)
        if result:
            print(result)
        portend.occupied('127.0.0.1', self.port, timeout=5)
        webtest.openURL('/ihopetheresnodefault', port=self.port)
        time.sleep(1)

    def stop(self):
        """Gracefully shutdown a server that is serving forever."""
        read_process(APACHE_PATH, '-k stop')