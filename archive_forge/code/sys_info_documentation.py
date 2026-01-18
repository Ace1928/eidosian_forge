from __future__ import absolute_import, division, print_function
import platform
from ansible.module_utils import distro
from ansible.module_utils.common._utils import get_all_subclasses

    Finds a subclass implementing desired functionality on the platform the code is running on

    :arg cls: Class to find an appropriate subclass for
    :returns: A class that implements the functionality on this platform

    Some Ansible modules have different implementations depending on the platform they run on.  This
    function is used to select between the various implementations and choose one.  You can look at
    the implementation of the Ansible :ref:`User module<user_module>` module for an example of how to use this.

    This function replaces ``basic.load_platform_subclass()``.  When you port code, you need to
    change the callers to be explicit about instantiating the class.  For instance, code in the
    Ansible User module changed from::

    .. code-block:: python

        # Old
        class User:
            def __new__(cls, args, kwargs):
                return load_platform_subclass(User, args, kwargs)

        # New
        class User:
            def __new__(cls, *args, **kwargs):
                new_cls = get_platform_subclass(User)
                return super(cls, new_cls).__new__(new_cls)
    