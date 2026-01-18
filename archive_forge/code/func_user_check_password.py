from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
def user_check_password(self, user, password):
    """
        Passlib replacement for User.check_password()
        """
    if password is None:
        return False
    hash = user.password
    if not self.is_password_usable(hash):
        return False
    cat = self.get_user_category(user)
    try:
        ok, new_hash = self.context.verify_and_update(password, hash, category=cat)
    except exc.UnknownHashError:
        return False
    if ok and new_hash is not None:
        user.password = new_hash
        user.save()
    return ok