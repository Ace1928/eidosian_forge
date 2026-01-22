from typing import Tuple, Union
import zope.interface
class IProxyInfo(zope.interface.Interface):
    """
    Data container for PROXY protocol header data.
    """
    header = zope.interface.Attribute('The raw byestring that represents the PROXY protocol header.')
    source = zope.interface.Attribute('An L{twisted.internet.interfaces.IAddress} representing the connection source.')
    destination = zope.interface.Attribute('An L{twisted.internet.interfaces.IAddress} representing the connection destination.')