import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class PasswordProperty(StringProperty):
    """

    Hashed property whose original value can not be
    retrieved, but still can be compared.

    Works by storing a hash of the original value instead
    of the original value.  Once that's done all that
    can be retrieved is the hash.

    The comparison

       obj.password == 'foo'

    generates a hash of 'foo' and compares it to the
    stored hash.

    Underlying data type for hashing, storing, and comparing
    is boto.utils.Password.  The default hash function is
    defined there ( currently sha512 in most cases, md5
    where sha512 is not available )

    It's unlikely you'll ever need to use a different hash
    function, but if you do, you can control the behavior
    in one of two ways:

      1) Specifying hashfunc in PasswordProperty constructor

         import hashlib

         class MyModel(model):
             password = PasswordProperty(hashfunc=hashlib.sha224)

      2) Subclassing Password and PasswordProperty

         class SHA224Password(Password):
             hashfunc=hashlib.sha224

         class SHA224PasswordProperty(PasswordProperty):
             data_type=MyPassword
             type_name="MyPassword"

         class MyModel(Model):
             password = SHA224PasswordProperty()

    """
    data_type = Password
    type_name = 'Password'

    def __init__(self, verbose_name=None, name=None, default='', required=False, validator=None, choices=None, unique=False, hashfunc=None):
        """
           The hashfunc parameter overrides the default hashfunc in boto.utils.Password.

           The remaining parameters are passed through to StringProperty.__init__"""
        super(PasswordProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)
        self.hashfunc = hashfunc

    def make_value_from_datastore(self, value):
        p = self.data_type(value, hashfunc=self.hashfunc)
        return p

    def get_value_for_datastore(self, model_instance):
        value = super(PasswordProperty, self).get_value_for_datastore(model_instance)
        if value and len(value):
            return str(value)
        else:
            return None

    def __set__(self, obj, value):
        if not isinstance(value, self.data_type):
            p = self.data_type(hashfunc=self.hashfunc)
            p.set(value)
            value = p
        super(PasswordProperty, self).__set__(obj, value)

    def __get__(self, obj, objtype):
        return self.data_type(super(PasswordProperty, self).__get__(obj, objtype), hashfunc=self.hashfunc)

    def validate(self, value):
        value = super(PasswordProperty, self).validate(value)
        if isinstance(value, self.data_type):
            if len(value) > 1024:
                raise ValueError('Length of value greater than maxlength')
        else:
            raise TypeError('Expecting %s, got %s' % (type(self.data_type), type(value)))