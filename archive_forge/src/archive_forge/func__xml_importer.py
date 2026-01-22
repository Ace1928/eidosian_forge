import os.path
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.dataportal import TableData
def _xml_importer():
    try:
        from lxml import etree
        return etree
    except ImportError:
        pass
    try:
        import xml.etree.cElementTree as etree
        return etree
    except ImportError:
        pass
    import xml.etree.ElementTree as etree
    return etree