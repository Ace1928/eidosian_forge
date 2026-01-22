from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InventorySoftwarePackage(_messages.Message):
    """Software package information of the operating system.

  Fields:
    aptPackage: Details of an APT package. For details about the apt package
      manager, see https://wiki.debian.org/Apt.
    cosPackage: Details of a COS package.
    googetPackage: Details of a Googet package. For details about the googet
      package manager, see https://github.com/google/googet.
    qfePackage: Details of a Windows Quick Fix engineering package. See
      https://docs.microsoft.com/en-
      us/windows/win32/cimwin32prov/win32-quickfixengineering for info in
      Windows Quick Fix Engineering.
    windowsApplication: Details of Windows Application.
    wuaPackage: Details of a Windows Update package. See
      https://docs.microsoft.com/en-us/windows/win32/api/_wua/ for information
      about Windows Update.
    yumPackage: Yum package info. For details about the yum package manager,
      see https://access.redhat.com/documentation/en-
      us/red_hat_enterprise_linux/6/html/deployment_guide/ch-yum.
    zypperPackage: Details of a Zypper package. For details about the Zypper
      package manager, see https://en.opensuse.org/SDB:Zypper_manual.
    zypperPatch: Details of a Zypper patch. For details about the Zypper
      package manager, see https://en.opensuse.org/SDB:Zypper_manual.
  """
    aptPackage = _messages.MessageField('InventoryVersionedPackage', 1)
    cosPackage = _messages.MessageField('InventoryVersionedPackage', 2)
    googetPackage = _messages.MessageField('InventoryVersionedPackage', 3)
    qfePackage = _messages.MessageField('InventoryWindowsQuickFixEngineeringPackage', 4)
    windowsApplication = _messages.MessageField('InventoryWindowsApplication', 5)
    wuaPackage = _messages.MessageField('InventoryWindowsUpdatePackage', 6)
    yumPackage = _messages.MessageField('InventoryVersionedPackage', 7)
    zypperPackage = _messages.MessageField('InventoryVersionedPackage', 8)
    zypperPatch = _messages.MessageField('InventoryZypperPatch', 9)