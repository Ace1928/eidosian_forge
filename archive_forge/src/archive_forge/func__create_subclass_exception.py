from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def _create_subclass_exception(self):

    class VimSubClass(exceptions.VimException):
        pass
    return VimSubClass