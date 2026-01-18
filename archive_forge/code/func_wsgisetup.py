import os
import re
import cherrypy
from cherrypy.test import helper
def wsgisetup(req):
    global loaded
    if not loaded:
        loaded = True
        options = req.get_options()
        cherrypy.config.update({'log.error_file': os.path.join(curdir, 'test.log'), 'environment': 'test_suite', 'server.socket_host': options['socket_host']})
        modname = options['testmod']
        mod = __import__(modname, globals(), locals(), [''])
        mod.setup_server()
        cherrypy.server.unsubscribe()
        cherrypy.engine.start()
    from mod_python import apache
    return apache.OK