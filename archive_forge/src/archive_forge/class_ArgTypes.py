import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class ArgTypes(object):

    def assertEqual(self, a, b):
        if not a == b:
            raise AssertionError('%s != %s' % (a, b))

    def assertIsInstance(self, value, v_type):
        assert isinstance(value, v_type), '%s is not instance of type %s' % (value, v_type)

    @expose(wsme.types.bytes)
    @validate(wsme.types.bytes)
    def setbytes(self, value):
        print(repr(value))
        self.assertEqual(type(value), wsme.types.bytes)
        return value

    @expose(wsme.types.text)
    @validate(wsme.types.text)
    def settext(self, value):
        print(repr(value))
        self.assertEqual(type(value), wsme.types.text)
        return value

    @expose(wsme.types.text)
    @validate(wsme.types.text)
    def settextnone(self, value):
        print(repr(value))
        self.assertEqual(type(value), type(None))
        return value

    @expose(bool)
    @validate(bool)
    def setbool(self, value):
        print(repr(value))
        self.assertEqual(type(value), bool)
        return value

    @expose(int)
    @validate(int)
    def setint(self, value):
        print(repr(value))
        self.assertEqual(type(value), int)
        return value

    @expose(float)
    @validate(float)
    def setfloat(self, value):
        print(repr(value))
        self.assertEqual(type(value), float)
        return value

    @expose(decimal.Decimal)
    @validate(decimal.Decimal)
    def setdecimal(self, value):
        print(repr(value))
        self.assertEqual(type(value), decimal.Decimal)
        return value

    @expose(datetime.date)
    @validate(datetime.date)
    def setdate(self, value):
        print(repr(value))
        self.assertEqual(type(value), datetime.date)
        return value

    @expose(datetime.time)
    @validate(datetime.time)
    def settime(self, value):
        print(repr(value))
        self.assertEqual(type(value), datetime.time)
        return value

    @expose(datetime.datetime)
    @validate(datetime.datetime)
    def setdatetime(self, value):
        print(repr(value))
        self.assertEqual(type(value), datetime.datetime)
        return value

    @expose(wsme.types.binary)
    @validate(wsme.types.binary)
    def setbinary(self, value):
        print(repr(value))
        self.assertEqual(type(value), bytes)
        return value

    @expose([wsme.types.bytes])
    @validate([wsme.types.bytes])
    def setbytesarray(self, value):
        print(repr(value))
        self.assertEqual(type(value), list)
        self.assertEqual(type(value[0]), wsme.types.bytes)
        return value

    @expose([wsme.types.text])
    @validate([wsme.types.text])
    def settextarray(self, value):
        print(repr(value))
        self.assertEqual(type(value), list)
        self.assertEqual(type(value[0]), wsme.types.text)
        return value

    @expose([datetime.datetime])
    @validate([datetime.datetime])
    def setdatetimearray(self, value):
        print(repr(value))
        self.assertEqual(type(value), list)
        self.assertEqual(type(value[0]), datetime.datetime)
        return value

    @expose(NestedOuter)
    @validate(NestedOuter)
    def setnested(self, value):
        print(repr(value))
        self.assertEqual(type(value), NestedOuter)
        return value

    @expose([NestedOuter])
    @validate([NestedOuter])
    def setnestedarray(self, value):
        print(repr(value))
        self.assertEqual(type(value), list)
        self.assertEqual(type(value[0]), NestedOuter)
        return value

    @expose({wsme.types.bytes: NestedOuter})
    @validate({wsme.types.bytes: NestedOuter})
    def setnesteddict(self, value):
        print(repr(value))
        self.assertEqual(type(value), dict)
        self.assertEqual(type(list(value.keys())[0]), wsme.types.bytes)
        self.assertEqual(type(list(value.values())[0]), NestedOuter)
        return value

    @expose(myenumtype)
    @validate(myenumtype)
    def setenum(self, value):
        print(value)
        self.assertEqual(type(value), wsme.types.bytes)
        return value

    @expose(NamedAttrsObject)
    @validate(NamedAttrsObject)
    def setnamedattrsobj(self, value):
        print(value)
        self.assertEqual(type(value), NamedAttrsObject)
        self.assertEqual(value.attr_1, 10)
        self.assertEqual(value.attr_2, 20)
        return value

    @expose(CustomObject)
    @validate(CustomObject)
    def setcustomobject(self, value):
        self.assertIsInstance(value, CustomObject)
        self.assertIsInstance(value.name, wsme.types.text)
        self.assertIsInstance(value.aint, int)
        return value

    @expose(ExtendedInt())
    @validate(ExtendedInt())
    def setextendedint(self, value):
        self.assertEqual(isinstance(value, ExtendedInt.basetype), True)
        return value