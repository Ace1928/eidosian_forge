import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QObject(ProxyClass):

    def metaObject(self):

        class _FakeMetaObject(object):

            def className(*args):
                return self.__class__.__name__
        return _FakeMetaObject()

    def objectName(self):
        return self._uic_name.split('.')[-1]

    def connect(cls, *args):
        slot_name = str(args[-1])
        if slot_name.endswith('.raise'):
            args = list(args[:-1])
            args.append(Literal(slot_name + '_'))
        ProxyClassMember(cls, 'connect', 0)(*args)
    connect = classmethod(connect)