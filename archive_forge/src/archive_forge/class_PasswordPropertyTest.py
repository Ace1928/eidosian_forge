import unittest
import logging
import time
class PasswordPropertyTest(unittest.TestCase):
    """Test the PasswordProperty"""

    def tearDown(self):
        cls = self.test_model()
        for obj in cls.all():
            obj.delete()

    def hmac_hashfunc(self):
        import hmac

        def hashfunc(msg):
            return hmac.new('mysecret', msg)
        return hashfunc

    def test_model(self, hashfunc=None):
        from boto.utils import Password
        from boto.sdb.db.model import Model
        from boto.sdb.db.property import PasswordProperty
        import hashlib

        class MyModel(Model):
            password = PasswordProperty(hashfunc=hashfunc)
        return MyModel

    def test_custom_password_class(self):
        from boto.utils import Password
        from boto.sdb.db.model import Model
        from boto.sdb.db.property import PasswordProperty
        import hmac, hashlib
        myhashfunc = hashlib.md5

        class MyPassword(Password):
            hashfunc = myhashfunc

        class MyPasswordProperty(PasswordProperty):
            data_type = MyPassword
            type_name = MyPassword.__name__

        class MyModel(Model):
            password = MyPasswordProperty()
        obj = MyModel()
        obj.password = 'bar'
        expected = myhashfunc('bar').hexdigest()
        log.debug('\npassword=%s\nexpected=%s' % (obj.password, expected))
        self.assertTrue(obj.password == 'bar')
        obj.save()
        id = obj.id
        time.sleep(5)
        obj = MyModel.get_by_id(id)
        self.assertEquals(obj.password, 'bar')
        self.assertEquals(str(obj.password), expected)

    def test_aaa_default_password_property(self):
        cls = self.test_model()
        obj = cls(id='passwordtest')
        obj.password = 'foo'
        self.assertEquals('foo', obj.password)
        obj.save()
        time.sleep(5)
        obj = cls.get_by_id('passwordtest')
        self.assertEquals('foo', obj.password)

    def test_password_constructor_hashfunc(self):
        import hmac
        myhashfunc = lambda msg: hmac.new('mysecret', msg)
        cls = self.test_model(hashfunc=myhashfunc)
        obj = cls()
        obj.password = 'hello'
        expected = myhashfunc('hello').hexdigest()
        self.assertEquals(obj.password, 'hello')
        self.assertEquals(str(obj.password), expected)
        obj.save()
        id = obj.id
        time.sleep(5)
        obj = cls.get_by_id(id)
        log.debug('\npassword=%s' % obj.password)
        self.assertTrue(obj.password == 'hello')