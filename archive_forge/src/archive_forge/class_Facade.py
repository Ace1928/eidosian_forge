from suds import *
from logging import getLogger
class Facade(Object):

    def __init__(self, name):
        Object.__init__(self)
        md = self.__metadata__
        md.facade = name