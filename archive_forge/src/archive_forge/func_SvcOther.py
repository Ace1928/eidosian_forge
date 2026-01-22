import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
def SvcOther(self, control):
    from cherrypy import process
    process.bus.publish(control_codes.key_for(control))