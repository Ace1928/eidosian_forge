import os.path
import cherrypy

Tutorial - Sessions

Storing session data in CherryPy applications is very easy: cherrypy
provides a dictionary called "session" that represents the session
data for the current user. If you use RAM based sessions, you can store
any kind of object into that dictionary; otherwise, you are limited to
objects that can be pickled.
