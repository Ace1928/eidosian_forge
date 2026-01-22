import cherrypy
import io
import logging
import os
import re
import sys
from more_itertools import always_iterable
import cherrypy
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil
Native adapter for serving CherryPy via mod_python

Basic usage:

##########################################
# Application in a module called myapp.py
##########################################

import cherrypy

class Root:
    @cherrypy.expose
    def index(self):
        return 'Hi there, Ho there, Hey there'


# We will use this method from the mod_python configuration
# as the entry point to our application
def setup_server():
    cherrypy.tree.mount(Root())
    cherrypy.config.update({'environment': 'production',
                            'log.screen': False,
                            'show_tracebacks': False})

##########################################
# mod_python settings for apache2
# This should reside in your httpd.conf
# or a file that will be loaded at
# apache startup
##########################################

# Start
DocumentRoot "/"
Listen 8080
LoadModule python_module /usr/lib/apache2/modules/mod_python.so

<Location "/">
    PythonPath "sys.path+['/path/to/my/application']"
    SetHandler python-program
    PythonHandler cherrypy._cpmodpy::handler
    PythonOption cherrypy.setup myapp::setup_server
    PythonDebug On
</Location>
# End

The actual path to your mod_python.so is dependent on your
environment. In this case we suppose a global mod_python
installation on a Linux distribution such as Ubuntu.

We do set the PythonPath configuration setting so that
your application can be found by from the user running
the apache2 instance. Of course if your application
resides in the global site-package this won't be needed.

Then restart apache2 and access http://127.0.0.1:8080
