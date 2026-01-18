import unittest
import logging
import time
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