import os.path
import cherrypy

Tutorial - Object inheritance

You are free to derive your request handler classes from any base
class you wish. In most real-world applications, you will probably
want to create a central base class used for all your pages, which takes
care of things like printing a common page header and footer.
